import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanorl.sync import SGLangWeightSync


class SyncTests(unittest.TestCase):
    def test_reload_failure_keeps_generation_paused(self):
        class Actor:
            def export_hf(self, path):
                path.mkdir()

        def post(endpoint, payload):
            if endpoint == "update_weights_from_disk":
                raise RuntimeError("reload failed")
            return {}

        with tempfile.TemporaryDirectory() as directory:
            sync = SGLangWeightSync("http://localhost:30000", Path(directory))
            with patch.object(sync, "_post", side_effect=post) as post_mock:
                with patch.object(sync, "_get", return_value={}):
                    with self.assertRaisesRegex(RuntimeError, "reload failed"):
                        sync.publish(Actor(), 1)
            self.assertNotIn("continue_generation", [call.args[0] for call in post_mock.call_args_list])

    def test_version_mismatch_keeps_generation_paused(self):
        class Actor:
            def export_hf(self, path):
                path.mkdir()

        with tempfile.TemporaryDirectory() as directory:
            sync = SGLangWeightSync("http://localhost:30000", Path(directory))
            with patch.object(sync, "_post", return_value={"success": True}) as post_mock:
                with patch.object(sync, "_get", return_value={"weight_version": "0"}):
                    with self.assertRaisesRegex(RuntimeError, "expected 1"):
                        sync.publish(Actor(), 1)
            self.assertNotIn("continue_generation", [call.args[0] for call in post_mock.call_args_list])
            self.assertIsNone(sync.committed_version)

    def test_export_then_pause_flush_reload_resume(self):
        events = []

        class Actor:
            def export_hf(self, path):
                events.append(("export", path.name))
                path.mkdir()

        with tempfile.TemporaryDirectory() as directory:
            sync = SGLangWeightSync("http://localhost:30000", Path(directory))
            def fake_get(endpoint):
                events.append((endpoint, None))
                return {"weight_version": "1"} if endpoint == "model_info" else {}

            def fake_post(endpoint, payload):
                events.append((endpoint, None))
                return {"success": True}

            with patch.object(sync, "_post", side_effect=fake_post):
                with patch.object(sync, "_get", side_effect=fake_get):
                    sync.publish(Actor(), 1)
            self.assertEqual(sync.committed_version, 1)
            with self.assertRaises(ValueError):
                sync.publish(Actor(), 1)

        self.assertEqual(
            [name for name, _ in events],
            ["export", "pause_generation", "flush_cache", "update_weights_from_disk", "model_info", "continue_generation"],
        )

    def test_missing_reload_ack_does_not_commit_or_resume(self):
        class Actor:
            def export_hf(self, path):
                path.mkdir()

        with tempfile.TemporaryDirectory() as directory:
            sync = SGLangWeightSync("http://localhost", Path(directory))
            with patch.object(sync, "_post", return_value={}) as post:
                with patch.object(sync, "_get", return_value={"weight_version": "1"}):
                    with self.assertRaisesRegex(RuntimeError, "explicitly confirm"):
                        sync.publish(Actor(), 1)
            self.assertIsNone(sync.committed_version)
            self.assertNotIn("continue_generation", [call.args[0] for call in post.call_args_list])


if __name__ == "__main__":
    unittest.main()
