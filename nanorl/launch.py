"""Single-node, two physical GPUs; only terminate process groups we created."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

from .config import parse_args
from .environment import require_single_rank


def select_gpus(train: str, rollout: str, inventory: str) -> tuple[str, str]:
    devices = {}
    for line in inventory.splitlines():
        index, gpu_uuid = (part.strip() for part in line.split(",", 1))
        devices[index] = devices[gpu_uuid] = gpu_uuid
    if train not in devices or rollout not in devices:
        raise ValueError("TRAIN_GPU/ROLLOUT_GPU must each be one nvidia-smi index or full GPU UUID (no lists/MIG)")
    if devices[train] == devices[rollout]:
        raise ValueError("Training and rollout must use different physical GPUs")
    return devices[train], devices[rollout]


def check_port(port: int):
    if not 1 <= port <= 65535:
        raise ValueError("Port must be in 1..65535")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def stop_process_group(process, timeout=10):
    # Also kill children when their group leader has already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                process.poll()  # Reap the leader; workers may still be shutting down.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
    process.wait()


def wait_ready(process, base_url, model_path, api_key, timeout):
    session = requests.Session()
    session.trust_env = False
    session.headers["Authorization"] = f"Bearer {api_key}"
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("SGLang exited during startup; inspect sglang.log")
            try:
                health = session.get(base_url + "/health", timeout=2)
                if health.ok:
                    # Health may be public; also require this run's model/instance identity.
                    info = session.get(base_url + "/model_info", timeout=2)
                    info.raise_for_status()
                    body = info.json()
                    if Path(body.get("model_path", "")).resolve() != model_path.resolve():
                        raise RuntimeError("SGLang model identity mismatch")
                    if body.get("served_model_name") != f"nanorl-{api_key}":
                        raise RuntimeError("SGLang instance identity mismatch; refusing an unrelated server")
                    if process.poll() is not None:
                        raise RuntimeError("SGLang exited before training started")
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.25)
        raise TimeoutError("SGLang startup timed out; inspect sglang.log")
    finally:
        session.close()


def main():
    require_single_rank()
    # The shell wrapper supplies the same trainer CLI, but this entry owns the server URL.
    config = parse_args()
    port = int(os.environ.get("PORT", "30000"))
    master_port = int(os.environ.get("MASTER_PORT", "29501"))
    if port == master_port:
        raise ValueError("PORT and MASTER_PORT must differ")
    check_port(port)
    check_port(master_port)
    expected_url = f"http://127.0.0.1:{port}"
    if config.sglang_url != expected_url:
        raise ValueError("Launcher owns the SGLang endpoint; set PORT instead of --sglang-url")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    train_gpu, rollout_gpu = select_gpus(os.environ.get("TRAIN_GPU", "0"), os.environ.get("ROLLOUT_GPU", "1"), inventory)
    run_dir = config.output_dir.resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    api_key = uuid.uuid4().hex
    common = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port),
                  NANORL_SGLANG_API_KEY=api_key, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    train_env, rollout_env = dict(common, CUDA_VISIBLE_DEVICES=train_gpu), dict(common, CUDA_VISIBLE_DEVICES=rollout_gpu)
    timeout = float(os.environ.get("STARTUP_TIMEOUT", "300"))
    if not 0 < timeout <= 3600:
        raise ValueError("STARTUP_TIMEOUT must be finite and in (0, 3600]")
    processes = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for role, env in (("train", train_env), ("rollout", rollout_env)):
            proc = subprocess.Popen([sys.executable, "-m", "nanorl.environment", "--model-path", str(config.model_path),
                                     "--role", role, "--max-sequence-length", str(config.max_sequence_length),
                                     "--weight-sync", config.weight_sync,
                                     "--actor-backend", config.actor_backend, "--actor-dtype", config.actor_dtype,
                                     "--report", str(run_dir / f"environment-{role}.json")], env=env, start_new_session=True)
            processes.append(proc)
            if proc.wait(timeout=timeout):
                raise RuntimeError(f"{role} preflight failed; see environment-{role}.json")
            stop_process_group(proc)
            processes.pop()
        (run_dir / "launch.json").write_text(json.dumps({"train_gpu": train_gpu, "rollout_gpu": rollout_gpu,
                                                       "port": port, "master_port": master_port, "model": str(config.model_path.resolve()),
                                                       "weight_sync": config.weight_sync, "rollout_dtype": config.rollout_dtype,
                                                       "actor_backend": config.actor_backend, "actor_dtype": config.actor_dtype,
                                                       "seed": config.seed, "deterministic_inference": True}, indent=2))
        # Recheck after preflight; UUID-scoped workers and a per-run API key prevent accidental reuse.
        check_port(port)
        with (run_dir / "sglang.log").open("a") as log:
            log.write(f"\n--- SGLang start {time.strftime('%Y-%m-%d %H:%M:%S')} seed={config.seed} ---\n")
            log.flush()
            server = subprocess.Popen([sys.executable, "-m", "sglang.launch_server", "--model-path", str(config.model_path),
                                       "--host", "127.0.0.1", "--port", str(port), "--tp", "1", "--api-key", api_key,
                                       "--served-model-name", f"nanorl-{api_key}",
                                       "--dtype", config.rollout_dtype,
                                       "--random-seed", str(config.seed), "--enable-deterministic-inference",
                                       "--context-length", str(config.max_sequence_length)],
                                      env=rollout_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            wait_ready(server, expected_url, config.model_path, api_key, timeout)
            trainer = subprocess.Popen([sys.executable, "-m", "nanorl.train", *sys.argv[1:]], env=train_env, start_new_session=True)
            processes.append(trainer)
            while trainer.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError("SGLang died during training; stopping trainer")
                time.sleep(0.25)
            if trainer.returncode:
                raise RuntimeError(f"Trainer exited with status {trainer.returncode}")
    finally:
        # A second Ctrl-C must not interrupt cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        for process in reversed(processes):
            stop_process_group(process)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
