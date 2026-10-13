"""Fail before model allocation, and record the actual runtime rather than guessed pins."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .conversion import HFCheckpoint


def require_single_rank():
    for name, expected in {"WORLD_SIZE": 1, "LOCAL_WORLD_SIZE": 1, "RANK": 0, "LOCAL_RANK": 0}.items():
        if int(os.environ.get(name, expected)) != expected:
            raise ValueError(f"{name} must be {expected}; torchrun/multiple training ranks are unsupported")


def check_model(path: Path, max_sequence_length: int):
    from transformers import AutoTokenizer

    checkpoint = HFCheckpoint(path)
    c = checkpoint.config
    # Explicit first target: Qwen2.5-0.5B(-Instruct), not arbitrary large Qwen models.
    target = {"hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24,
              "num_attention_heads": 14, "num_key_value_heads": 2, "vocab_size": 151936}
    if any(c[key] != value for key, value in target.items()) or not c.get("tie_word_embeddings"):
        raise ValueError("The supported target is Qwen2.5-0.5B-Instruct with tied embeddings")
    if not 2 <= max_sequence_length <= c["max_position_embeddings"]:
        raise ValueError("Requested context length is outside the model's supported range")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": "1+1?"}], tokenize=True, add_generation_prompt=True)
    if not ids or min(ids) < 0 or max(ids) >= c["vocab_size"] or len(tokenizer) > c["vocab_size"]:
        raise ValueError("Tokenizer/chat template is incompatible with the model vocabulary")
    return {"path": str(path.resolve()), "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
            "tensor_count": len(checkpoint.locations), "max_position_embeddings": c["max_position_embeddings"]}


def runtime_snapshot() -> dict:
    packages = {}
    for name in ("torch", "transformers", "sglang", "megatron-core", "megatron-bridge", "transformer-engine", "safetensors", "requests"):
        try:
            dist = metadata.distribution(name)
            packages[name] = {"version": dist.version, "direct_url": json.loads(dist.read_text("direct_url.json") or "null")}
        except metadata.PackageNotFoundError:
            packages[name] = {"missing": True}
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "container_image": os.environ.get("NANORL_IMAGE", "unrecorded"), "packages": packages,
            "pip_freeze": freeze.stdout.splitlines(), "pip_freeze_error": freeze.stderr if freeze.returncode else None}


def preflight(model_path: Path, role: str, max_sequence_length: int, weight_sync: str = "disk",
              actor_backend: str = "torch", actor_dtype: str = "float32") -> dict:
    require_single_rank()
    if sys.version_info < (3, 10) or platform.system() != "Linux":
        raise RuntimeError("Runtime requires Linux and Python >= 3.10; local CPU unit tests are separate")
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not dist.is_nccl_available():
        raise RuntimeError("Each role requires exactly one visible CUDA GPU and NCCL")
    report = runtime_snapshot()
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version,memory.total", "--format=csv,noheader"], text=True)
    report.update(role=role, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), nvidia_smi=driver.splitlines(),
                  model=check_model(model_path, max_sequence_length))
    if role == "train":
        if actor_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Training GPU does not support BF16")
        report.update(actor_backend=actor_backend, actor_dtype=actor_dtype)
        import transformer_engine.pytorch
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        inspect.signature(AutoBridge.from_hf_pretrained).bind(str(model_path), trust_remote_code=False)
        inspect.signature(AutoBridge.to_megatron_provider).bind(None, load_weights=False)
        inspect.signature(AutoBridge.load_hf_weights).bind(None, [], str(model_path))
        inspect.signature(AutoBridge.save_hf_pretrained).bind(None, [], path="unused")
        if actor_backend == "megatron":
            from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
            from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
            from megatron.core.transformer.module import Float16Module
            inspect.signature(DistributedDataParallel).bind(config=None, ddp_config=None, module=None, disable_bucketing=True)
            inspect.signature(DistributedDataParallelConfig).bind(grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                                                                 overlap_param_gather=False, use_distributed_optimizer=False,
                                                                 check_for_nan_in_grad=False)
            inspect.signature(OptimizerConfig).bind(optimizer="adam", lr=1e-6, min_lr=1e-6, weight_decay=0.0,
                                                    bf16=True, fp16=False, params_dtype=torch.bfloat16,
                                                    use_distributed_optimizer=False, clip_grad=1.0, loss_scale=None,
                                                    adam_beta1=.9, adam_beta2=.999, adam_eps=1e-8)
            inspect.signature(get_megatron_optimizer).bind(config=None, model_chunks=[], use_gloo_process_groups=False)
            inspect.signature(Float16Module).bind(config=None, module=None)
            inspect.signature(finalize_model_grads).bind([])
        if weight_sync == "nccl":
            from sglang.srt.utils import init_custom_process_group
            inspect.signature(AutoBridge.export_hf_weights).bind(None, [], cpu=False)
            inspect.signature(init_custom_process_group).bind(backend="nccl", init_method="tcp://127.0.0.1:1",
                                                              world_size=2, rank=0, group_name="preflight", timeout=None)
        inspect.signature(GPTModelProvider.provide_distributed_model).bind(None, wrap_with_ddp=False, bf16=False, fp16=False)
        for field in ("gradient_accumulation_fusion", "params_dtype", "pipeline_dtype", "autocast_dtype"):
            if field not in GPTModelProvider.__dataclass_fields__:
                raise RuntimeError(f"Incompatible Bridge provider: missing {field}")
        report["bridge_source_sha256"] = hashlib.sha256(Path(inspect.getfile(AutoBridge)).read_bytes()).hexdigest()
    else:
        from sglang.srt.entrypoints.http_server import app
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.server_args import ServerArgs

        inspect.signature(SamplingParams).bind_partial(sampling_seed=42)
        inspect.signature(ServerArgs).bind_partial(enable_deterministic_inference=True, random_seed=42)

        required = {("GET", "/health"), ("GET", "/model_info"), ("POST", "/generate"),
                    ("POST", "/pause_generation"), ("GET", "/flush_cache"),
                    ("POST", "/continue_generation")}
        if weight_sync == "nccl":
            required |= {("POST", "/init_weights_update_group"), ("POST", "/update_weights_from_distributed"),
                         ("POST", "/destroy_weights_update_group")}
        else:
            required.add(("POST", "/update_weights_from_disk"))
        routes = {(method, route.path) for route in app.routes for method in (getattr(route, "methods", None) or [])}
        if required - routes:
            raise RuntimeError(f"Incompatible SGLang HTTP API: missing {sorted(required - routes)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--role", choices=["train", "rollout"], default="train")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weight-sync", choices=["disk", "nccl"], default="disk")
    parser.add_argument("--actor-backend", choices=["megatron", "torch"], default="torch")
    parser.add_argument("--actor-dtype", choices=["bfloat16", "float32"], default="float32")
    args = parser.parse_args()
    report = {"status": "failed"}
    try:
        report.update(preflight(args.model_path, args.role, args.max_sequence_length, args.weight_sync,
                                args.actor_backend, args.actor_dtype), status="passed")
    except Exception as exc:
        report.update(runtime_snapshot(), error=f"{type(exc).__name__}: {exc}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.report), "error": report.get("error")}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
