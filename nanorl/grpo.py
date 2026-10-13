from __future__ import annotations

from collections.abc import Sequence
import math


def group_advantages(rewards: Sequence[float], group_size: int, eps: float = 1e-6, group_ids=None) -> list[float]:
    """Sample-standard-deviation normalization; explicit IDs support reordered batches."""
    if group_size < 2 or len(rewards) == 0 or len(rewards) % group_size:
        raise ValueError("Rewards must contain complete groups of at least two")
    if not math.isfinite(eps) or eps <= 0 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Rewards and positive eps must be finite")
    if group_ids is None:
        group_ids = [i // group_size for i in range(len(rewards))]
    if len(group_ids) != len(rewards):
        raise ValueError("Group IDs and rewards must align")
    groups = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError("Every group must be complete")
        group = [rewards[index] for index in indices]
        mean = sum(group) / group_size
        variance = sum((x - mean) ** 2 for x in group) / (group_size - 1)
        std = variance**0.5
        for index in indices:
            advantages[index] = (rewards[index] - mean) / (std + eps)
    if not all(math.isfinite(x) for x in advantages):
        raise ValueError("Non-finite advantages")
    return advantages


def clipped_policy_loss(new_log_probs, old_log_probs, advantage: float, clip_ratio: float):
    """One response's token-mean GRPO surrogate; tensors use torch operations."""
    import torch

    if new_log_probs.ndim != 1 or new_log_probs.shape != old_log_probs.shape or new_log_probs.numel() == 0:
        raise ValueError("New and old log-probability vectors must match and be nonempty")
    if not math.isfinite(advantage) or not 0 < clip_ratio < 1:
        raise ValueError("Finite advantage and clip_ratio in (0, 1) are required")
    if not torch.isfinite(new_log_probs).all() or not torch.isfinite(old_log_probs).all():
        raise ValueError("Non-finite logprobs")
    ratio = torch.exp(new_log_probs - old_log_probs)
    if not torch.isfinite(ratio).all():
        raise ValueError("Non-finite policy ratio")
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()
