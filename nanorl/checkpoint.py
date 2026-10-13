"""Synchronous single-rank training checkpoints, distinct from HF inference exports."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from .run_log import config_dict


# Runtime knobs (microbatch size, transport, concurrency, paths, total steps) may change.
TRAINING_FIELDS = ("prompts_per_step", "group_size", "max_new_tokens", "max_sequence_length",
                   "learning_rate", "clip_ratio", "actor_backend", "actor_dtype", "rollout_dtype",
                   "reward", "seed", "shuffle")


def model_metadata(model_path: Path):
    """Hash model semantics and tokenization; weights themselves are in the checkpoint."""
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
             "tokenizer.model", "chat_template.jinja")
    return {name: hashlib.sha256((model_path / name).read_bytes()).hexdigest()
            for name in names if (model_path / name).is_file()}


def seed_training(seed: int):
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Called after actor construction, when Megatron model parallel groups exist.
    model_parallel_cuda_manual_seed(seed)


def rng_state():
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "megatron": get_cuda_rng_tracker().get_states()}


def restore_rng(state):
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    get_cuda_rng_tracker().set_states(state["megatron"])


def atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Make the rename durable on the supported Linux local filesystem.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(directory: Path, *, session_id: str, step: int, actor, sampler, config, metadata):
    import torch

    if not actor.valid or not 0 <= actor.policy_version <= step:
        raise RuntimeError("Cannot checkpoint invalid or inconsistent actor state")
    state = {"format_version": 1, "step": step, "policy_version": actor.policy_version,
             "models": [model.state_dict() for model in actor.models],
             # Includes BF16's FP32 master parameters, Adam moments and optimizer step.
             "optimizer": actor.optimizer.state_dict(), "rng": rng_state(),
             "sampler": sampler.state_dict(), "config": config_dict(config),
             "model_metadata": metadata, "session_id": session_id}
    path = directory / f"{session_id}-step-{step:08d}.pt"
    atomic_write(path, lambda stream: torch.save(state, stream))
    pointer = json.dumps({"file": path.name, "step": step, "policy_version": actor.policy_version}).encode()
    atomic_write(directory / "latest.json", lambda stream: stream.write(pointer))
    return path


def load_checkpoint(path: Path, *, actor, sampler, config, metadata):
    import torch

    if path.is_dir():
        path = path / "latest.json"
    if path.suffix == ".json":
        filename = json.loads(path.read_text())["file"]
        if Path(filename).name != filename:
            raise ValueError("Checkpoint pointer must name a file in its own directory")
        path = path.parent / filename
    # Only load trusted nanorl training artifacts: RNG/optimizer state uses pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format_version"] != 1:
        raise ValueError("Unsupported nanorl checkpoint format")
    if state["model_metadata"] != metadata:
        raise ValueError("Resume model configuration or tokenizer differs")
    current = config_dict(config)
    changed = [key for key in TRAINING_FIELDS if state["config"][key] != current[key]]
    if changed:
        raise ValueError(f"Resume training configuration differs: {', '.join(changed)}")
    step, version = state["step"], state["policy_version"]
    if type(step) is not int or type(version) is not int or not 0 <= version <= step <= config.steps:
        raise ValueError("Invalid saved step/version, or --steps is below the completed step")
    if len(state["models"]) != len(actor.models):
        raise ValueError("Resume model chunk count differs")
    sampler.load_state_dict(state["sampler"])
    for model, saved in zip(actor.models, state["models"]):
        model.load_state_dict(saved, strict=True)
    actor.optimizer.load_state_dict(state["optimizer"])
    # Do NOT reload master params from BF16: that would discard saved FP32 precision.
    if actor.native_training is not None:
        actor.native_training.zero_grad()
    else:
        actor.optimizer.zero_grad(set_to_none=True)
    actor.policy_version = version
    if not all(torch.isfinite(parameter).all() for model in actor.models for parameter in model.parameters()):
        actor.valid = False
        raise ValueError("Non-finite checkpoint model parameters")
    return step, state["rng"], path.resolve()
