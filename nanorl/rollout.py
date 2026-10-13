from __future__ import annotations

import math
import os
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests


@dataclass(frozen=True)
class Sample:
    prompt_ids: list[int]
    response_ids: list[int]
    old_log_probs: list[float]
    response_text: str
    sample_id: str
    group_id: str
    policy_version: int
    finish_reason: str

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def validate(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id or not isinstance(self.group_id, str) or not self.group_id or type(self.policy_version) is not int or self.policy_version < 0:
            raise ValueError("Sample needs IDs and a nonnegative policy version")
        if not isinstance(self.response_text, str):
            raise ValueError("Response text must be a string")
        if not self.prompt_ids or not self.response_ids or any(type(x) is not int or x < 0 for x in self.prompt_ids + self.response_ids):
            raise ValueError("Prompt and response must contain nonnegative integer token IDs")
        if len(self.response_ids) != len(self.old_log_probs) or not all(math.isfinite(x) for x in self.old_log_probs):
            raise ValueError("Response tokens and finite logprobs must align")
        if self.finish_reason not in {"stop", "length"}:
            raise ValueError("Only completed stop/length responses can train")


def validate_batch(samples, group_size: int, policy_version: int) -> None:
    if group_size < 2 or not samples:
        raise ValueError("A batch needs nonempty groups of at least two")
    groups, ids = {}, set()
    for sample in samples:
        sample.validate()
        if sample.sample_id in ids or sample.policy_version != policy_version:
            raise ValueError("Duplicate sample ID or stale policy version")
        ids.add(sample.sample_id)
        groups.setdefault(sample.group_id, []).append(sample)
    for group in groups.values():
        if len(group) != group_size or any(s.prompt_ids != group[0].prompt_ids for s in group):
            raise ValueError("Each group needs exactly group_size responses to the same prompt")


def prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )
    if not ids:
        raise ValueError("Prompt produced no tokens")
    return list(ids)


class SGLangRollout:
    def __init__(self, base_url: str, max_new_tokens: int, max_sequence_length: int, *, seed: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.max_sequence_length = max_sequence_length
        self.seed = seed

    def generate_batch(self, prompt_groups, *, group_size: int, policy_version: int, concurrency: int):
        """Bound concurrent HTTP requests; preserve logical order despite completion order."""
        if concurrency < 1 or group_size < 2 or not prompt_groups:
            raise ValueError("Need positive concurrency and nonempty groups of at least two")
        jobs = [(ids, group_id, f"{group_id}/response-{i}")
                for group_id, ids in prompt_groups for i in range(group_size)]

        def generate_one(job):
            ids, group_id, sample_id = job
            return self.generate(ids, group_id=group_id, sample_id=sample_id, policy_version=policy_version)

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="rollout") as executor:
            # A failure discards the entire batch; no incomplete group reaches reward/update.
            samples = list(executor.map(generate_one, jobs))
        validate_batch(samples, group_size, policy_version)
        return samples

    def require_version(self, policy_version: int) -> None:
        key = os.environ.get("NANORL_SGLANG_API_KEY")
        response = requests.get(f"{self.base_url}/model_info", timeout=30,
                                headers={"Authorization": f"Bearer {key}"} if key else {})
        response.raise_for_status()
        if str(response.json().get("weight_version")) != str(policy_version):
            raise RuntimeError(f"SGLang is not serving expected policy version {policy_version}")

    def generate(self, prompt_ids: list[int], *, sample_id: str, group_id: str, policy_version: int) -> Sample:
        if not sample_id or not group_id or type(policy_version) is not int or policy_version < 0:
            raise ValueError("Generation requires sample/group IDs and a committed policy version")
        if not prompt_ids or any(type(x) is not int or x < 0 for x in prompt_ids):
            raise ValueError("Prompt must contain valid token IDs")
        budget = min(self.max_new_tokens, self.max_sequence_length - len(prompt_ids))
        if budget < 1:
            raise ValueError("Prompt already fills the context window")
        self.require_version(policy_version)
        sampling_params = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": budget}
        if self.seed is not None:
            # Stable per logical request, independent of thread scheduling and process hash seed.
            digest = hashlib.sha256(f"{self.seed}:{sample_id}".encode()).digest()
            sampling_params["sampling_seed"] = int.from_bytes(digest[:4], "big") % (2**31)
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            },
            timeout=180,
            headers={"Authorization": f"Bearer {os.environ['NANORL_SGLANG_API_KEY']}"} if os.environ.get("NANORL_SGLANG_API_KEY") else {},
        )
        response.raise_for_status()
        data = response.json()
        meta = data["meta_info"]
        entries = meta["output_token_logprobs"]
        ids = [entry[1] for entry in entries]
        log_probs = [float(entry[0]) for entry in entries]
        reason = meta.get("finish_reason", {}).get("type")
        if type(meta.get("completion_tokens")) is not int or meta["completion_tokens"] != len(ids) or len(ids) > budget:
            raise ValueError("SGLang completion count disagrees with returned token logprobs or budget")
        if reason == "length" and len(ids) != budget:
            raise ValueError("Length termination occurred before the requested token budget")
        if "output_ids" in data and data["output_ids"] != ids:
            raise ValueError("SGLang output_ids disagree with logprob token IDs")
        if "weight_version" in meta and str(meta["weight_version"]) != str(policy_version):
            raise RuntimeError("Response came from a different policy version")
        sample = Sample(list(prompt_ids), ids, log_probs, data["text"], sample_id, group_id, policy_version, reason)
        sample.validate()
        self.require_version(policy_version)
        return sample
