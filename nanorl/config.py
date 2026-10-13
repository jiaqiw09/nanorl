from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: Path
    data_path: Path
    output_dir: Path
    sglang_url: str
    steps: int = 3
    prompts_per_step: int = 2
    group_size: int = 4
    max_new_tokens: int = 32
    max_sequence_length: int = 256
    learning_rate: float = 1e-6
    clip_ratio: float = 0.2
    weight_sync: str = "disk"
    rollout_dtype: str = "bfloat16"
    sync_bucket_mb: int = 64
    sync_timeout: int = 120
    actor_backend: str = "megatron"
    actor_dtype: str = "bfloat16"
    rollout_concurrency: int = 4
    reward: str = "exact-integer"
    train_microbatch_size: int = 1
    seed: int = 42
    shuffle: bool = True
    checkpoint_every: int = 1
    resume: Path | None = None

    def validate(self) -> None:
        if not self.model_path.is_dir():
            raise ValueError(f"Model directory does not exist: {self.model_path}")
        if not self.data_path.is_file():
            raise ValueError(f"Dataset does not exist: {self.data_path}")
        if self.steps < 1 or self.prompts_per_step < 1 or self.group_size < 2:
            raise ValueError("steps and prompts_per_step must be positive; group_size must be >= 2")
        if self.max_new_tokens < 1 or self.max_sequence_length < 2:
            raise ValueError("Invalid sequence length")
        if self.max_new_tokens >= self.max_sequence_length:
            raise ValueError("max_new_tokens must be smaller than max_sequence_length")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not 0 < self.clip_ratio < 1:
            raise ValueError("learning_rate must be positive and clip_ratio must be between 0 and 1")
        if not self.sglang_url.startswith(("http://", "https://")):
            raise ValueError("sglang_url must be an HTTP URL")
        if self.weight_sync not in {"disk", "nccl"} or self.rollout_dtype not in {"bfloat16", "float32"}:
            raise ValueError("Unsupported weight sync backend or rollout dtype")
        if self.sync_bucket_mb < 1 or self.sync_timeout < 1:
            raise ValueError("sync_bucket_mb and sync_timeout must be positive")
        if self.actor_backend not in {"torch", "megatron"} or self.actor_dtype not in {"float32", "bfloat16"}:
            raise ValueError("Unsupported actor backend or dtype")
        if self.actor_backend == "torch" and self.actor_dtype != "float32":
            raise ValueError("--actor-backend torch requires --actor-dtype float32")
        if self.rollout_concurrency < 1:
            raise ValueError("rollout_concurrency must be positive")
        if self.train_microbatch_size < 1 or self.checkpoint_every < 1:
            raise ValueError("train_microbatch_size and checkpoint_every must be positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.resume is not None and not self.resume.exists():
            raise ValueError(f"Resume checkpoint does not exist: {self.resume}")
        if self.reward not in {"exact-integer", "last-integer"} and ":" not in self.reward:
            raise ValueError("Reward must name a built-in or module:function")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Two-GPU SGLang + Megatron GRPO demo")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("examples/math.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo/weights"))
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--weight-sync", choices=["disk", "nccl"], default="disk")
    parser.add_argument("--rollout-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--sync-bucket-mb", type=int, default=64)
    parser.add_argument("--sync-timeout", type=int, default=120)
    parser.add_argument("--actor-backend", choices=["megatron", "torch"], default="megatron")
    parser.add_argument("--actor-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--rollout-concurrency", type=int, default=4)
    parser.add_argument("--reward", default="exact-integer")
    parser.add_argument("--train-microbatch-size", type=int, default=1, help="Responses per actor forward/backward")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Cycle through dataset in file order")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save every N completed steps, and the final step")
    parser.add_argument("--resume", type=Path, help="Training .pt, latest.json, or checkpoints directory")
    config = Config(**vars(parser.parse_args()))
    config.validate()
    return config
