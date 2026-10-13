from __future__ import annotations

from pathlib import Path
import os
import time

import requests


class SGLangWeightSync:
    """Export full HF weights and ask a separate SGLang server to reload them."""

    def __init__(self, base_url: str, output_dir: Path):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir
        key = os.environ.get("NANORL_SGLANG_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.committed_version = None
        self.timeout = 300
        self.last_metrics = {}

    def _post(self, endpoint: str, payload: dict) -> dict:
        response = requests.post(f"{self.base_url}/{endpoint}", json=payload, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        result = response.json() if "application/json" in response.headers.get("Content-Type", "") else {}
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"SGLang {endpoint} failed: {result}")
        return result

    def _get(self, endpoint: str) -> dict:
        response = requests.get(f"{self.base_url}/{endpoint}", timeout=self.timeout, headers=self.headers)
        response.raise_for_status()
        return response.json() if "application/json" in response.headers.get("Content-Type", "") else {}

    def publish(self, actor, version: int) -> Path:
        started = time.monotonic()
        if type(version) is not int or version < 0 or (self.committed_version is not None and version <= self.committed_version):
            raise ValueError("Published versions must be nonnegative and strictly increasing")
        checkpoint = self.output_dir / f"weight_v{version:06d}"
        actor.export_hf(checkpoint)
        # Generation is synchronous in this demo, so no requests are in flight here.
        self._post("pause_generation", {})
        # A failed/timed-out reload can leave engine state uncertain. Abort the
        # training loop and keep serving paused until the engine is recovered.
        self._get("flush_cache")
        result = self._post(
            "update_weights_from_disk",
            {"model_path": str(checkpoint.resolve()), "weight_version": str(version)},
        )
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError("SGLang reload did not explicitly confirm success; generation remains paused")
        actual = self._get("model_info").get("weight_version")
        if str(actual) != str(version):
            raise RuntimeError(f"SGLang reports weight version {actual}, expected {version}")
        self._post("continue_generation", {})
        self.committed_version = version
        self.last_metrics = {"sync_backend": "disk", "sync_seconds": time.monotonic() - started}
        return checkpoint

    def close(self):
        pass
