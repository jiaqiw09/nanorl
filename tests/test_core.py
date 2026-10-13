import importlib.util
import unittest
from unittest.mock import patch

from nanorl.grpo import clipped_policy_loss, group_advantages
from nanorl.reward import ExactIntegerReward, Example
from nanorl.rollout import SGLangRollout


class CoreTests(unittest.TestCase):
    def test_group_advantages_are_per_prompt(self):
        values = group_advantages([0, 1, 1, 0, 0, 0], group_size=3)
        self.assertAlmostEqual(sum(values[:3]), 0)
        self.assertEqual(values[3:], [0, 0, 0])
        self.assertLess(values[0], 0)
        self.assertGreater(values[1], 0)

    def test_integer_reward_requires_exact_answer(self):
        reward = ExactIntegerReward()
        example = Example("2 + 3?", 5)
        self.assertEqual(reward(example, " 5 "), 1.0)
        self.assertEqual(reward(example, "The answer is 5"), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_clip_blocks_improving_ratio_beyond_bound(self):
        import torch

        current = torch.tensor([0.0], requires_grad=True)
        old = torch.tensor([-1.0])
        loss = clipped_policy_loss(current, old, advantage=1.0, clip_ratio=0.2)
        self.assertAlmostEqual(loss.item(), -1.2)
        loss.backward()
        self.assertEqual(current.grad.item(), 0.0)

    @patch("nanorl.rollout.requests.post")
    def test_rollout_keeps_token_logprob_alignment(self, post):
        post.return_value.json.return_value = {
            "text": "5",
            "meta_info": {"output_token_logprobs": [[-0.2, 42, None], [-0.1, 2, None]],
                          "completion_tokens": 2, "finish_reason": {"type": "stop"}},
        }
        rollout = SGLangRollout("http://localhost:30000", 8, 16)
        with patch.object(rollout, "require_version"):
            sample = rollout.generate([10, 11], sample_id="s", group_id="g", policy_version=0)
        self.assertEqual(sample.response_ids, [42, 2])
        self.assertEqual(sample.old_log_probs, [-0.2, -0.1])
        self.assertEqual(post.call_args.kwargs["json"]["input_ids"], [10, 11])


if __name__ == "__main__":
    unittest.main()
