"""Native Megatron DDP + optimizer for the single-rank actor."""
from __future__ import annotations

import math


class MegatronTraining:
    def __init__(self, model, learning_rate: float, dtype: str):
        import torch
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
        from megatron.core.transformer.module import Float16Module

        self.raw_model = model
        bf16 = dtype == "bfloat16"
        if bf16:
            model = Float16Module(config=model.config, module=model)
        # Keep synchronization explicit: all response microsteps accumulate before finalization.
        # DP=1 does not need sharded optimizer state or communication overlap.
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=False,  # Check explicitly; no Megatron training rerun state machine.
        )
        self.model = DistributedDataParallel(config=model.config, ddp_config=ddp_config,
                                             module=model, disable_bucketing=True)
        config = OptimizerConfig(
            optimizer="adam", lr=learning_rate, min_lr=learning_rate, weight_decay=0.0,
            bf16=bf16, fp16=False, params_dtype=torch.bfloat16 if bf16 else torch.float32,
            use_distributed_optimizer=False, clip_grad=1.0, loss_scale=None,
            adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        )
        config.timers = None
        self.optimizer = get_megatron_optimizer(config=config, model_chunks=[self.model],
                                                use_gloo_process_groups=False)
        self.did_backward = False

    def zero_grad(self):
        # Clearing optimizer grads alone leaves the DDP accumulation buffers dirty.
        self.model.zero_grad_buffer()
        self.optimizer.zero_grad()
        self.did_backward = False

    def backward(self, loss):
        # BF16 uses no dynamic loss scale, but use the native interface consistently.
        self.optimizer.scale_loss(loss).backward()
        self.did_backward = True

    def step(self) -> dict:
        import torch
        from megatron.core.distributed import finalize_model_grads

        if not self.did_backward:
            raise RuntimeError("No backward microstep was executed")
        for name, parameter in self.raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            grad = getattr(parameter, "main_grad", None)
            if grad is None or grad.dtype != torch.float32:
                raise RuntimeError(f"Expected Megatron FP32 main_grad for {name}")
            if not torch.isfinite(grad).all():
                raise ValueError(f"Non-finite Megatron gradient for {name}")
        # Completes DDP gradient synchronization and model-specific gradient finalization.
        # num_tokens is omitted: actor already applies token-mean then response-mean loss.
        finalize_model_grads([self.model])
        # Native optimizer owns clipping, BF16's FP32 master params and the copy back.
        # Calling torch.clip_grad_norm_ on the BF16 model's .grad here would be incorrect.
        successful, grad_norm, _ = self.optimizer.step()
        if not successful:
            return {"update_applied": False, "grad_norm": None, "optimizer_skipped": True}
        if grad_norm is None or not math.isfinite(float(grad_norm)):
            raise ValueError("Megatron optimizer returned an invalid gradient norm")
        return {"update_applied": True, "grad_norm": float(grad_norm), "optimizer_skipped": False}
