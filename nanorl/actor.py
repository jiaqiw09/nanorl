from __future__ import annotations

import os
import math
from pathlib import Path


def response_log_probs_from_logits(logits, tokens, prompt_length: int, vocab_size: int):
    """Causal shift for one unpadded sequence; exclude padded vocabulary rows."""
    import torch

    if logits.ndim != 3 or logits.shape[:2] != tokens.shape or tokens.shape[0] != 1:
        raise ValueError("Expected logits [1, sequence, vocab] aligned with tokens")
    if not 0 < prompt_length < tokens.shape[1] or not 0 < vocab_size <= logits.shape[-1]:
        raise ValueError("Invalid response span or vocabulary size")
    if torch.any(tokens < 0) or torch.any(tokens >= vocab_size):
        raise ValueError("Token outside the original HF vocabulary")
    selected = logits[:, prompt_length - 1 : -1, :vocab_size].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite response logits")
    targets = tokens[:, prompt_length:]
    return selected.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1)


class MegatronActor:
    """Single-rank Megatron Core actor, initialized with Megatron Bridge."""

    def __init__(self, model_path: Path, learning_rate: float, *, backend: str = "torch", dtype: str = "float32"):
        import torch
        import torch.distributed as dist
        from .conversion import QwenConversion
        from .environment import check_model, require_single_rank

        require_single_rank()
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if backend not in {"torch", "megatron"} or dtype not in {"float32", "bfloat16"}:
            raise ValueError("Unsupported actor backend or dtype")
        if backend == "torch" and dtype != "float32":
            raise ValueError("The torch baseline requires FP32; use the Megatron backend for BF16")
        if not torch.cuda.is_available():
            raise RuntimeError("Megatron training requires a CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one training GPU with CUDA_VISIBLE_DEVICES")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected training GPU does not support BF16")
        check_model(model_path, 2)
        torch.cuda.set_device(0)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        self._owns_process_group = not dist.is_initialized()
        if dist.is_initialized() and (dist.get_world_size() != 1 or dist.get_rank() != 0):
            raise RuntimeError("Existing distributed group must have one rank")
        if self._owns_process_group:
            dist.init_process_group("nccl", rank=0, world_size=1)
        try:
            self.backend, self.dtype = backend, dtype
            self.conversion = QwenConversion(model_path, dtype=dtype)
            self.models = self.conversion.load()
            self.model = self.models[0]
            self.native_training = None
            if backend == "megatron":
                from .megatron_backend import MegatronTraining
                self.native_training = MegatronTraining(self.model, learning_rate, dtype)
                self.model = self.native_training.model
                self.optimizer = self.native_training.optimizer
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.0)
            self.model.train()
            self.policy_version = 0
            self.valid = True
        except BaseException:
            self.close()
            raise

    def response_log_probs(self, prompt_ids: list[int], response_ids: list[int]):
        import torch

        if not prompt_ids or not response_ids:
            raise ValueError("Prompt and response must both contain tokens")
        vocab_size = self.conversion.source.config["vocab_size"]
        if any(type(x) is not int or not 0 <= x < vocab_size for x in prompt_ids + response_ids):
            raise ValueError("Token outside the original HF vocabulary")
        if len(prompt_ids) + len(response_ids) > self.conversion.source.config["max_position_embeddings"]:
            raise ValueError("Sequence exceeds model context length")
        device = next(self.model.parameters()).device
        tokens = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
        logits = self.model(input_ids=tokens, position_ids=positions, attention_mask=None)
        if isinstance(logits, tuple):
            logits = logits[0]
        return response_log_probs_from_logits(logits, tokens, len(prompt_ids), vocab_size)

    def batch_response_log_probs(self, samples):
        from .batching import collate_responses, response_log_probs

        config = self.conversion.source.config
        tokens, positions, mask = collate_responses(
            samples, vocab_size=config["vocab_size"], max_length=config["max_position_embeddings"],
            device=next(self.model.parameters()).device,
        )
        # Qwen's attention is causal. Right padding is strictly after all valid tokens;
        # valid queries cannot attend to it. Keep the native causal attention kernel.
        logits = self.model(input_ids=tokens, position_ids=positions, attention_mask=None)
        if isinstance(logits, tuple):
            logits = logits[0]
        flat = response_log_probs(logits, tokens, mask, config["vocab_size"])
        return flat.split([len(s.response_ids) for s in samples])

    def update(self, samples, advantages: list[float], clip_ratio: float, group_size: int,
               microbatch_size: int = 1) -> dict:
        import torch

        from .grpo import clipped_policy_loss
        from .rollout import validate_batch

        if not self.valid:
            raise RuntimeError("Actor invalidated by a failed update; reload from a known checkpoint")
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        validate_batch(samples, group_size, self.policy_version)
        if len(samples) != len(advantages) or not all(math.isfinite(x) for x in advantages) or not 0 < clip_ratio < 1:
            raise ValueError("Samples need aligned finite advantages and clip_ratio in (0, 1)")
        native = getattr(self, "native_training", None)
        if native is not None:
            native.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)
        metrics = {"policy_loss": 0.0, "grad_norm": 0.0, "update_applied": False, "policy_version": self.policy_version,
                   "actor_backend": getattr(self, "backend", "torch"), "actor_dtype": getattr(self, "dtype", "float32"),
                   "optimizer_skipped": False, "microbatches": 0,
                   "train_microbatch_size": microbatch_size, "ratio_mean": None, "clip_fraction": None}
        # AdamW momentum can move parameters even when all current gradients are zero.
        # With no group signal, skip the optimizer and keep the published version.
        if not any(advantages):
            return metrics
        try:
            parameters = list(self.model.parameters())
            if not all(torch.isfinite(p).all() for p in parameters):
                raise ValueError("Non-finite actor parameter before update")
            ratio_sum, clipped_tokens, response_tokens = 0.0, 0, 0
            for start in range(0, len(samples), microbatch_size):
                microbatch = samples[start : start + microbatch_size]
                # Preserve the single-response entry point used by the FP32 baseline tools.
                currents = ([self.response_log_probs(microbatch[0].prompt_ids, microbatch[0].response_ids)]
                            if len(microbatch) == 1 else self.batch_response_log_probs(microbatch))
                losses = []
                for sample, current, advantage in zip(microbatch, currents, advantages[start : start + microbatch_size]):
                    old = torch.tensor(sample.old_log_probs, dtype=current.dtype, device=current.device)
                    losses.append(clipped_policy_loss(current, old, advantage, clip_ratio))
                    ratio = (current.detach() - old).exp()
                    ratio_sum += ratio.sum().item()
                    clipped_tokens += ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).sum().item()
                    response_tokens += ratio.numel()
                # Global sequence denominator, including a short final microbatch.
                loss = torch.stack(losses).sum() / len(samples)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite policy loss")
                if native is not None:
                    native.backward(loss)
                else:
                    loss.backward()
                metrics["policy_loss"] += loss.detach().item()
                metrics["microbatches"] += 1
            metrics.update(ratio_mean=ratio_sum / response_tokens, clip_fraction=clipped_tokens / response_tokens)
            if not math.isfinite(metrics["policy_loss"]):
                raise ValueError("Non-finite accumulated loss")
            if native is not None:
                metrics.update(native.step())
            else:
                if not any(p.grad is not None for p in parameters):
                    raise ValueError("Actor produced no gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                metrics["grad_norm"] = norm.item()
                self.optimizer.step()
                metrics["update_applied"] = True
            if not all(torch.isfinite(p).all() for p in parameters):
                raise ValueError("Non-finite actor parameter after optimizer step")
            if metrics["update_applied"]:
                self.policy_version += 1
            metrics["policy_version"] = self.policy_version
            return metrics
        except BaseException:
            self.valid = False
            if native is not None:
                native.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)
            raise

    def export_hf(self, path: Path) -> None:
        if not self.valid:
            raise RuntimeError("Cannot publish an actor after a failed update")
        self.conversion.export(self.models, path)

    def iter_hf_weights(self):
        if not self.valid:
            raise RuntimeError("Cannot publish an actor after a failed update")
        yield from self.conversion.iter_hf_weights(self.models)

    def close(self):
        import torch.distributed as dist
        from megatron.core import parallel_state

        if self._owns_process_group and dist.is_initialized():
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
