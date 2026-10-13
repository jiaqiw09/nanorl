"""One training GPU -> one SGLang GPU, with a separate two-rank NCCL group."""
from __future__ import annotations

import socket
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta

from .sync import SGLangWeightSync


class NCCLWeightSync(SGLangWeightSync):
    def __init__(self, base_url, output_dir, *, dtype="bfloat16", bucket_mb=64, timeout=120):
        super().__init__(base_url, output_dir)
        self.dtype = dtype
        self.bucket_bytes = bucket_mb * 1024 * 1024
        self.timeout = timeout
        self.group_name = f"nanorl-{uuid.uuid4().hex}"
        self.group = None
        self.failed = False
        self.remote_requested = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sglang-weight-receiver")

    @staticmethod
    def _require_success(result, operation):
        if not isinstance(result, dict) or result.get("success") is not True:
            raise RuntimeError(f"SGLang did not confirm {operation}: {result}")

    def _connect(self):
        from sglang.srt.utils import init_custom_process_group

        # The training WORLD remains size one. dist.new_group() cannot add SGLang's rank.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.remote_requested = True
        receiver = self.executor.submit(self._post, "init_weights_update_group", {
            "master_address": "127.0.0.1", "master_port": port, "rank_offset": 1,
            "world_size": 2, "group_name": self.group_name, "backend": "nccl",
        })
        self.group = init_custom_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
                                               world_size=2, rank=0, group_name=self.group_name,
                                               timeout=timedelta(seconds=self.timeout))
        self._require_success(receiver.result(timeout=self.timeout), "NCCL group initialization")

    def _buckets(self, actor):
        import torch

        bucket, size = [], 0
        # Closing the generator restores the actor's mode even when a send fails midway.
        with closing(actor.iter_hf_weights()) as weights:
            for name, weight in weights:
                tensor = weight.to(dtype=getattr(torch, self.dtype)).contiguous()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite weight after transport dtype conversion: {name}")
                nbytes = tensor.numel() * tensor.element_size()
                if bucket and size + nbytes > self.bucket_bytes:
                    yield bucket, False
                    bucket, size = [], 0
                bucket.append((name, tensor))
                size += nbytes
            # Exhausting Bridge's iterator also checks that every required parameter was emitted.
            if not bucket:
                raise ValueError("No weights to synchronize")
            yield bucket, True

    def publish(self, actor, version: int):
        import torch.distributed as dist

        if self.failed:
            raise RuntimeError("Failed NCCL synchronizer cannot be reused; restart both roles")
        if type(version) is not int or version < 0 or version != actor.policy_version:
            raise ValueError("Requested weight version must match the actor")
        if self.committed_version is not None and version <= self.committed_version:
            raise ValueError("Published versions must increase")
        started = time.monotonic()
        transferred, buckets = 0, 0
        try:
            if self.group is None:
                self._connect()
            self._post("pause_generation", {})
            self._get("flush_cache")
            with closing(self._buckets(actor)) as stream:
                for bucket, final in stream:
                    payload = {"names": [name for name, _ in bucket],
                               "dtypes": [str(t.dtype).removeprefix("torch.") for _, t in bucket],
                               "shapes": [list(t.shape) for _, t in bucket],
                               "group_name": self.group_name, "flush_cache": False}
                    if final:
                        payload["weight_version"] = str(version)
                    # HTTP blocks while SGLang receives. Start it before matching NCCL broadcasts.
                    receiver = self.executor.submit(self._post, "update_weights_from_distributed", payload)
                    handles = []
                    for _, tensor in bucket:
                        if receiver.done():
                            self._require_success(receiver.result(), "weight receive")
                        handles.append(dist.broadcast(tensor, src=0, group=self.group, async_op=True))
                        transferred += tensor.numel() * tensor.element_size()
                    for handle in handles:
                        handle.wait(timeout=timedelta(seconds=self.timeout))
                    self._require_success(receiver.result(timeout=self.timeout), "weight receive")
                    buckets += 1
            self._get("flush_cache")
            actual = self._get("model_info").get("weight_version")
            if str(actual) != str(version):
                raise RuntimeError(f"SGLang reports version {actual}, expected {version}")
            self._post("continue_generation", {})
            self.committed_version = version
            self.last_metrics = {"sync_backend": "nccl", "sync_seconds": time.monotonic() - started,
                                 "sync_bytes": transferred, "sync_buckets": buckets, "sync_dtype": self.dtype}
            return None  # No checkpoint file is created by this transport.
        except BaseException:
            # Partially updated SGLang remains paused; never fall back to a different transport mid-update.
            self.failed = True
            raise

    def close(self):
        import torch.distributed as dist

        try:
            receiver = None
            if self.remote_requested:
                receiver = self.executor.submit(self._post, "destroy_weights_update_group", {"group_name": self.group_name})
            if self.group is not None:
                dist.destroy_process_group(self.group)
                self.group = None
            if receiver is not None:
                self._require_success(receiver.result(timeout=self.timeout), "NCCL group destruction")
        except Exception as exc:
            warnings.warn(f"NCCL cleanup failed; launcher will terminate its workers: {exc}")
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
