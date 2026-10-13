from __future__ import annotations

import json
import re
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: int


class Reward(Protocol):
    def __call__(self, example: Example, response: str) -> float: ...


class ExactIntegerReward:
    """A deterministic outcome reward, intentionally easy to replace."""

    def __call__(self, example: Example, response: str) -> float:
        match = re.fullmatch(r"\s*([+-]?\d+)\s*", response)
        return float(match is not None and int(match.group(1)) == example.answer)


class LastIntegerReward:
    """Allow explanation text; compare its last standalone integer with the answer."""

    def __call__(self, example: Example, response: str) -> float:
        numbers = re.findall(r"(?<![\w.])[+-]?\d+(?![\w.])", response)
        return float(bool(numbers) and int(numbers[-1]) == example.answer)


def load_reward(name: str) -> Reward:
    if name == "exact-integer":
        return ExactIntegerReward()
    if name == "last-integer":
        return LastIntegerReward()
    if ":" not in name:
        raise ValueError("Reward must be exact-integer, last-integer, or module:function")
    module_name, function_name = name.split(":", 1)
    reward = getattr(importlib.import_module(module_name), function_name)
    if not callable(reward):
        raise ValueError(f"Reward {name} is not callable")
    return reward


def load_examples(path: Path) -> list[Example]:
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("prompt"), str) or type(item.get("answer")) is not int:
            raise ValueError(f"Invalid prompt/answer on line {line_number}")
        examples.append(Example(item["prompt"], item["answer"]))
    if not examples:
        raise ValueError("Dataset is empty")
    return examples
