import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from nanorl.grpo import group_advantages
from nanorl.rollout import Sample, SGLangRollout, validate_batch


class SampleTests(unittest.TestCase):
    def sample(self, sample_id="a", group_id="g"):
        return Sample([1], [2], [-0.1], "5", sample_id, group_id, 0, "stop")

    def test_groups_reorder_without_changing_advantages(self):
        advantages = group_advantages([0, 0, 1, 0], 2, group_ids=["a", "b", "a", "b"])
        self.assertAlmostEqual(advantages[0], -1 / (math.sqrt(2) + 2e-6))
        self.assertEqual(advantages[1], 0)
        self.assertEqual(advantages[3], 0)
        self.assertAlmostEqual(advantages[0], -advantages[2])

    def test_invalid_groups_and_rewards_fail(self):
        for rewards, ids in (([0, float("nan")], ["a", "a"]), ([0, float("inf")], ["a", "a"]),
                             ([0, 1], ["a", "b"]), ([0, 1], ["a"]), ([0], ["a"])):
            with self.assertRaises(ValueError):
                group_advantages(rewards, 2, group_ids=ids)

    def test_batch_contract(self):
        a, b = self.sample(), self.sample("b")
        validate_batch([a, b], 2, 0)
        for samples in ([a], [a, a], [a, replace(b, prompt_ids=[3])], [a, replace(b, policy_version=1)],
                        [a, replace(b, response_ids=[])], [a, replace(b, old_log_probs=[float("nan")])],
                        [a, replace(b, finish_reason="abort")]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                validate_batch(samples, 2, 0)

    @patch("nanorl.rollout.requests.get")
    @patch("nanorl.rollout.requests.post")
    def test_eos_truncation_counts_and_version_checks(self, post, get):
        get.return_value.json.return_value = {"weight_version": "0"}
        rollout = SGLangRollout("http://localhost", 2, 4)
        def generate(meta, output_ids=None):
            data = {"text": "5", "meta_info": meta}
            if output_ids is not None:
                data["output_ids"] = output_ids
            post.return_value.json.return_value = data
            return rollout.generate([1], sample_id="s", group_id="g", policy_version=0)
        stopped = {"output_token_logprobs": [[-0.2, 2, None]], "completion_tokens": 1,
                   "finish_reason": {"type": "stop", "matched": 2}}
        sample = generate(stopped)
        self.assertEqual(sample.response_ids, [2])  # EOS stays in the loss span.
        self.assertFalse(sample.truncated)
        truncated = dict(stopped, output_token_logprobs=[[-0.2, 2], [-0.3, 3]], completion_tokens=2,
                         finish_reason={"type": "length"})
        self.assertTrue(generate(truncated).truncated)
        for changes in ({"completion_tokens": 2}, {"output_token_logprobs": []},
                        {"finish_reason": {"type": "abort"}}, {"finish_reason": {"type": "length"}},
                        {"output_token_logprobs": [[float("nan"), 2]]}):
            with self.assertRaises(ValueError):
                generate(dict(stopped, **changes))
        with self.assertRaises(ValueError):
            generate(stopped, output_ids=[3])
        get.return_value.json.side_effect = [{"weight_version": "0"}, {"weight_version": "1"}]
        with self.assertRaisesRegex(RuntimeError, "expected policy"):
            generate(stopped)

    @patch("nanorl.rollout.requests.post")
    def test_empty_and_oversize_prompt_fail_before_request(self, post):
        rollout = SGLangRollout("http://localhost", 2, 4)
        for prompt in ([], [1] * 4):
            with self.assertRaises(ValueError):
                rollout.generate(prompt, sample_id="s", group_id="g", policy_version=0)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
