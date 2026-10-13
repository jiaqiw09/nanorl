import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from nanorl.config import Config
from nanorl.conversion import expected_shapes, read_config
from nanorl.environment import require_single_rank
from nanorl.launch import check_port, select_gpus, stop_process_group, wait_ready


class StartupTests(unittest.TestCase):
    def test_gpu_aliases_and_invalid_selections(self):
        inventory = "0, GPU-aaa\n1, GPU-bbb\n"
        self.assertEqual(select_gpus("0", "GPU-bbb", inventory), ("GPU-aaa", "GPU-bbb"))
        for train, rollout in (("0", "GPU-aaa"), ("0,1", "1"), ("2", "1"), ("MIG-aaa", "1")):
            with self.subTest(train=train, rollout=rollout), self.assertRaises(ValueError):
                select_gpus(train, rollout, inventory)

    def test_external_multi_rank_is_rejected(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
                require_single_rank()

    def test_port_conflict_and_invalid_range(self):
        # Mock only bind: actual socket binding is restricted on some local sandboxes.
        with patch("nanorl.launch.socket.socket") as socket:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError("address in use")
            with self.assertRaises(OSError):
                check_port(30000)
        for port in (0, -1, 65536):
            with self.assertRaises(ValueError):
                check_port(port)

    def test_server_ownership_and_startup_failure(self):
        server = Mock()
        server.poll.return_value = None
        with patch("nanorl.launch.requests.Session") as session_class:
            session = session_class.return_value
            session.headers = {}
            session.get.return_value.ok = True
            session.get.return_value.json.return_value = {"model_path": "/model", "served_model_name": "nanorl-secret"}
            wait_ready(server, "http://localhost:30000", Path("/model"), "secret", 1)
            self.assertEqual(session.headers["Authorization"], "Bearer secret")
            for body in ({"model_path": "/wrong", "served_model_name": "nanorl-secret"},
                         {"model_path": "/model", "served_model_name": "unrelated"}):
                session.get.return_value.json.return_value = body
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    wait_ready(server, "http://localhost:30000", Path("/model"), "secret", 1)
            server.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "exited"):
                wait_ready(server, "http://localhost:30000", Path("/model"), "secret", 1)
            server.poll.return_value = None
            with self.assertRaises(TimeoutError):
                wait_ready(server, "http://localhost:30000", Path("/model"), "secret", 0)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_cleanup_reaches_worker_process(self):
        # Child writes an acknowledgement on SIGTERM. Tests the actual OS group, not mocked kill().
        with tempfile.TemporaryDirectory() as directory:
            ack = Path(directory) / "worker-stopped"
            child = ("import signal,time,pathlib,sys; "
                     "signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).write_text('stopped'), sys.exit(0))); "
                     "print('ready', flush=True); time.sleep(30)")
            parent = ("import subprocess,sys,time; "
                      "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],stdout=subprocess.PIPE,text=True); "
                      "print(p.stdout.readline(),flush=True); time.sleep(30)")
            process = subprocess.Popen([sys.executable, "-c", parent, child, str(ack)], start_new_session=True,
                                       stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(), "ready")
                stop_process_group(process, timeout=1)
                self.assertIsNotNone(process.poll())
                self.assertEqual(ack.read_text(), "stopped")
            finally:
                stop_process_group(process, timeout=0)
                process.stdout.close()

    def test_config_rejects_nonfinite_learning_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.touch()
            for value in (float("nan"), float("inf"), -1):
                with self.assertRaises(ValueError):
                    Config(root, data, root, "http://localhost", learning_rate=value).validate()

    def test_qwen_config_admission_and_gqa_shapes(self):
        config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 8,
                  "intermediate_size": 12, "num_hidden_layers": 1, "num_attention_heads": 4,
                  "num_key_value_heads": 2, "vocab_size": 10, "max_position_embeddings": 32}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config))
            self.assertEqual(expected_shapes(read_config(path))["model.layers.0.self_attn.k_proj.weight"], (4, 8))
            for changes in ({"model_type": "qwen3"}, {"num_key_value_heads": 3}, {"quantization_config": {"bits": 4}}):
                (path / "config.json").write_text(json.dumps(dict(config, **changes)))
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_roundtrip_failure_writes_failure_report(self):
        from nanorl.roundtrip import main

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "roundtrip"
            with patch.object(sys, "argv", ["roundtrip", "--model-path", "/missing", "--output-dir", str(output)]):
                with patch("nanorl.roundtrip.preflight", side_effect=RuntimeError("CUDA unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                        main()
            report = json.loads((output / "roundtrip.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stages"], {})


if __name__ == "__main__":
    unittest.main()
