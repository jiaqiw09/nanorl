"""Qwen2.5 dense conversion: Bridge executes it; independent checks audit it."""
from __future__ import annotations

import json
from pathlib import Path


def read_config(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen2" or config.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError("Only dense Qwen2ForCausalLM (Qwen2.5) checkpoints are supported")
    if config.get("quantization_config") or config.get("rope_scaling"):
        raise ValueError("Quantized weights and scaled RoPE are outside this demo's scope")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "max_position_embeddings"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Invalid model config: {key}")
    if config["hidden_size"] % config["num_attention_heads"] or config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("Invalid GQA head dimensions")
    return config


def expected_shapes(c: dict) -> dict:
    h, f, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    kv = c["num_key_value_heads"] * (h // c["num_attention_heads"])
    shapes = {"model.embed_tokens.weight": (v, h), "lm_head.weight": (v, h), "model.norm.weight": (h,)}
    for i in range(c["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        shapes.update({prefix + key: shape for key, shape in {
            "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
            "self_attn.q_proj.weight": (h, h), "self_attn.q_proj.bias": (h,),
            "self_attn.k_proj.weight": (kv, h), "self_attn.k_proj.bias": (kv,),
            "self_attn.v_proj.weight": (kv, h), "self_attn.v_proj.bias": (kv,),
            "self_attn.o_proj.weight": (h, h), "mlp.gate_proj.weight": (f, h),
            "mlp.up_proj.weight": (f, h), "mlp.down_proj.weight": (h, f),
        }.items()})
    return shapes


class HFCheckpoint:
    """Inspect all shard headers, then load only one tensor at a time on CPU."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self.path, self.config = path, read_config(path)
        index = path / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        files = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
        self.locations, self.metadata = {}, {}
        for filename in files:
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"Invalid shard filename: {filename}")
            with safe_open(str(path / filename), framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    if name in self.locations:
                        raise ValueError(f"Duplicate tensor: {name}")
                    view = shard.get_slice(name)
                    self.locations[name] = filename
                    self.metadata[name] = (tuple(view.get_shape()), view.get_dtype())
        if weight_map is not None and weight_map != self.locations:
            raise ValueError("Shard index does not match the tensors stored in the shards")
        shapes = expected_shapes(self.config)
        required = set(shapes)
        if self.config.get("tie_word_embeddings", False):
            required.remove("lm_head.weight")
        missing, extra = required - self.locations.keys(), self.locations.keys() - shapes.keys()
        if missing or extra:
            raise ValueError(f"Checkpoint keys: missing={sorted(missing)}, extra={sorted(extra)}")
        for name, (shape, dtype) in self.metadata.items():
            if shape != shapes[name] or dtype not in {"F32", "F16", "BF16"}:
                raise ValueError(f"Invalid tensor {name}: shape={shape}, dtype={dtype}; expected {shapes[name]}")
        if self.config.get("tie_word_embeddings", False) and "lm_head.weight" in self.locations:
            import torch
            if not torch.equal(self.tensor("lm_head.weight"), self.tensor("model.embed_tokens.weight")):
                raise ValueError("Tied lm_head and embedding contain different values")

    def tensor(self, name):
        from safetensors import safe_open

        if name == "lm_head.weight" and name not in self.locations and self.config.get("tie_word_embeddings", False):
            name = "model.embed_tokens.weight"
        with safe_open(str(self.path / self.locations[name]), framework="pt", device="cpu") as shard:
            return shard.get_tensor(name)


def compare_tensor(name, actual, expected, *, atol=0.0, rtol=0.0) -> dict:
    import torch

    if actual.shape != expected.shape:
        raise ValueError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{name}: non-finite weight")
    error = (actual - expected).abs()
    if not torch.all(error <= atol + rtol * expected.abs()):
        raise ValueError(f"{name}: max_abs_error={error.max().item():.9g}, atol={atol}, rtol={rtol}")
    return {"name": name, "max_abs_error": error.max().item(), "numel": actual.numel()}


def hf_views(name, tensor, c):
    """Independent inverse layout, used ONLY by the verifier, never by Bridge."""
    direct = {"embedding.word_embeddings.weight": "model.embed_tokens.weight",
              "output_layer.weight": "lm_head.weight", "decoder.final_layernorm.weight": "model.norm.weight"}
    if name in direct:
        if name != "decoder.final_layernorm.weight":
            tensor = tensor[:c["vocab_size"]]  # MCore may pad the vocabulary.
        return {direct[name]: tensor}
    parts = name.split(".")
    if len(parts) < 5 or parts[:2] != ["decoder", "layers"] or not parts[2].isdigit():
        raise ValueError(f"Unrecognized Megatron parameter: {name}")
    prefix, suffix = f"model.layers.{parts[2]}.", ".".join(parts[3:])
    direct_layer = {"self_attention.linear_proj.weight": "self_attn.o_proj.weight",
                    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
                    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
                    "mlp.linear_fc2.weight": "mlp.down_proj.weight"}
    if suffix in direct_layer:
        return {prefix + direct_layer[suffix]: tensor}
    if suffix in {"self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"}:
        groups, heads, h = c["num_key_value_heads"], c["num_attention_heads"], c["hidden_size"]
        tail = (h,) if suffix.endswith("weight") else ()
        grouped = tensor.reshape(groups, heads // groups + 2, h // heads, *tail)
        q, k, v = grouped.split([heads // groups, 1, 1], dim=1)
        kind = suffix.rsplit(".", 1)[1]
        return {prefix + f"self_attn.{label}_proj.{kind}": value.reshape(-1, *tail)
                for label, value in zip(("q", "k", "v"), (q, k, v))}
    if suffix == "mlp.linear_fc1.weight":
        gate, up = tensor.chunk(2, dim=0)
        return {prefix + "mlp.gate_proj.weight": gate, prefix + "mlp.up_proj.weight": up}
    raise ValueError(f"Unrecognized Megatron parameter: {name}")


def verify_megatron(model, source: HFCheckpoint) -> list:
    results, seen = [], set()
    tied = source.config.get("tie_word_embeddings", False)
    if bool(model.share_embeddings_and_output_weights) != tied:
        raise ValueError("Megatron tied embedding setting disagrees with HF config")
    if tied:
        output = getattr(model.output_layer, "weight", None)
        if output is not None and output.data_ptr() != model.embedding.word_embeddings.weight.data_ptr():
            raise ValueError("Megatron output and embedding are not actually shared")
    for name, param in model.named_parameters():
        if str(param.dtype) != "torch.float32":
            raise ValueError(f"Expected FP32 Megatron parameter: {name} is {param.dtype}")
        for hf_name, value in hf_views(name, param, source.config).items():
            if hf_name in seen:
                raise ValueError(f"Duplicate Megatron mapping: {hf_name}")
            results.append(compare_tensor(hf_name, value, source.tensor(hf_name)))
            seen.add(hf_name)
    required = set(expected_shapes(source.config))
    if tied:
        required.remove("lm_head.weight")
    if required - seen:
        raise ValueError(f"Missing Megatron mappings: {sorted(required - seen)}")
    return results


def compare_checkpoints(source: HFCheckpoint, exported: HFCheckpoint) -> list:
    # Serialization metadata/name/dtype may change; model semantics must not.
    keys = ("model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings",
            "rope_theta", "rms_norm_eps", "hidden_act", "tie_word_embeddings", "bos_token_id", "eos_token_id")
    for key in keys:
        if source.config.get(key) != exported.config.get(key):
            raise ValueError(f"Export changed config.{key}")
    results = []
    for name in expected_shapes(source.config):
        before, after = source.tensor(name), exported.tensor(name)
        # FP16/BF16 input -> FP32 training/export is a deliberate lossless widening.
        if after.dtype != before.dtype and str(after.dtype) != "torch.float32":
            raise ValueError(f"Unexpected dtype conversion for {name}: {before.dtype} -> {after.dtype}")
        row = compare_tensor(name, after, before)
        row.update(source_dtype=str(before.dtype), export_dtype=str(after.dtype))
        results.append(row)
    return results


class QwenConversion:
    def __init__(self, model_path: Path, dtype: str = "float32"):
        import torch
        from megatron.bridge import AutoBridge

        if dtype not in {"float32", "bfloat16"}:
            raise ValueError("Actor dtype must be float32 or bfloat16")
        params_dtype = getattr(torch, dtype)
        self.source = HFCheckpoint(model_path)
        self.bridge = AutoBridge.from_hf_pretrained(str(model_path), trust_remote_code=False)
        self.provider = self.bridge.to_megatron_provider(load_weights=False)
        for key, value in {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                           "bf16": dtype == "bfloat16", "fp16": False, "params_dtype": params_dtype,
                           "pipeline_dtype": params_dtype, "autocast_dtype": params_dtype,
                           "hidden_dropout": 0.0, "attention_dropout": 0.0,
                           "gradient_accumulation_fusion": False}.items():
            setattr(self.provider, key, value)

    def load(self):
        from megatron.core.utils import unwrap_model

        self.provider.finalize()
        # Do not pass bf16=False for a BF16 provider: Bridge may override all dtype fields.
        # Drop Bridge's precision wrapper; the training backend owns wrapping after loading.
        models = unwrap_model(self.provider.provide_distributed_model(
            wrap_with_ddp=False, bf16=self.provider.bf16, fp16=False))
        if len(models) != 1:
            raise RuntimeError("This demo supports one Megatron stage only")
        self.bridge.load_hf_weights(models, str(self.source.path))
        return models

    def iter_hf_weights(self, models):
        """Stream current GPU weights; no checkpoint write and no full CPU state dict."""
        import torch

        shapes = expected_shapes(self.source.config)
        required = set(shapes)
        if self.source.config.get("tie_word_embeddings", False):
            required.remove("lm_head.weight")
        seen, patched = set(), []
        modes = [model.training for model in models]
        try:
            for model in models:
                model.eval()
                if not hasattr(model.config, "share_embeddings_and_output_weights"):
                    model.config.share_embeddings_and_output_weights = model.share_embeddings_and_output_weights
                    patched.append(model.config)
            with torch.no_grad():
                for item in self.bridge.export_hf_weights(models, cpu=False):
                    # NVIDIA Bridge yields two fields; the slime Bridge fork adds a source name.
                    name, weight = item[0], item[1]
                    if name in seen or name not in shapes or tuple(weight.shape) != shapes[name]:
                        raise ValueError(f"Unexpected, duplicate or misshaped streamed HF tensor: {name}")
                    if not weight.is_cuda or not weight.is_floating_point() or not torch.isfinite(weight).all():
                        raise ValueError(f"Expected finite CUDA weights for {name}")
                    seen.add(name)
                    yield name, weight.detach()
            if required - seen:
                raise ValueError(f"Incomplete streamed weights: {sorted(required - seen)}")
        finally:
            for config in patched:
                delattr(config, "share_embeddings_and_output_weights")
            for model, training in zip(models, modes):
                model.train(training)

    def export(self, models, path: Path):
        import torch
        from transformers import AutoTokenizer

        path.mkdir(parents=True, exist_ok=False)
        modes = [model.training for model in models]
        # Some Core/Bridge combinations read this model attribute from config.
        # Match slime's narrow compatibility shim, restoring config afterwards.
        patched_configs = []
        try:
            for model in models:
                model.eval()
                if not hasattr(model.config, "share_embeddings_and_output_weights"):
                    model.config.share_embeddings_and_output_weights = model.share_embeddings_and_output_weights
                    patched_configs.append(model.config)
            with torch.no_grad():
                self.bridge.save_hf_pretrained(models, path=str(path))
            HFCheckpoint(path)  # Reject missing keys, shards, bad shapes and tied weights.
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if not tokenizer.chat_template:
                raise ValueError("Export is missing the chat template")
        finally:
            for config in patched_configs:
                delattr(config, "share_embeddings_and_output_weights")
            for model, training in zip(models, modes):
                model.train(training)
