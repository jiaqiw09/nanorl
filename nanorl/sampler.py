"""An endless epoch sampler whose next batch survives a training restart."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict


class ResumableSampler:
    def __init__(self, examples, *, seed: int, shuffle: bool):
        if not examples:
            raise ValueError("Cannot sample an empty dataset")
        self.examples = examples
        content = json.dumps([asdict(example) for example in examples], sort_keys=True, ensure_ascii=False)
        self.fingerprint = hashlib.sha256(content.encode()).hexdigest()
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.epoch, self.cursor = 0, 0
        self.order = list(range(len(examples)))
        if shuffle:
            self.rng.shuffle(self.order)

    def take(self, count: int):
        if count < 1:
            raise ValueError("Sample count must be positive")
        indices = []
        while len(indices) < count:
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self.order = list(range(len(self.examples)))
                if self.shuffle:
                    self.rng.shuffle(self.order)
            end = min(len(self.order), self.cursor + count - len(indices))
            indices.extend(self.order[self.cursor:end])
            self.cursor = end
        return [(index, self.examples[index]) for index in indices]

    def state_dict(self):
        return {"fingerprint": self.fingerprint, "shuffle": self.shuffle,
                "epoch": self.epoch, "cursor": self.cursor, "order": self.order.copy(),
                "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        if state["fingerprint"] != self.fingerprint or state["shuffle"] != self.shuffle:
            raise ValueError("Resume dataset content/order or shuffle setting differs")
        order, cursor, epoch = state["order"], state["cursor"], state["epoch"]
        if (len(order) != len(self.examples) or any(type(i) is not int for i in order)
                or sorted(order) != list(range(len(self.examples)))
                or type(cursor) is not int or not 0 <= cursor <= len(order)
                or type(epoch) is not int or epoch < 0):
            raise ValueError("Invalid saved sampler permutation, cursor or epoch")
        self.rng.setstate(state["rng"])
        self.order, self.cursor, self.epoch = order.copy(), cursor, epoch
