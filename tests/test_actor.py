import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanorl.actor import MegatronActor, response_log_probs_from_logits
from nanorl.grpo import clipped_policy_loss
from nanorl.rollout import Sample


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for actor numerical tests")
class ActorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def actor(self):
        torch = self.torch
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = torch.nn.Parameter(torch.zeros(4, 4))
            def forward(self, input_ids, position_ids, attention_mask):
                return self.table[input_ids]
        actor = MegatronActor.__new__(MegatronActor)
        actor.model = TinyModel()
        actor.conversion = SimpleNamespace(source=SimpleNamespace(config={"vocab_size": 4, "max_position_embeddings": 16}))
        actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.01, weight_decay=0)
        actor.valid, actor.policy_version = True, 0
        return actor

    def samples(self, actor, responses=([1], [2])):
        with self.torch.no_grad():
            return [Sample([0], response, actor.response_log_probs([0], response).tolist(), "", str(i), "g", actor.policy_version, "stop")
                    for i, response in enumerate(responses)]

    def test_shift_single_token_and_padded_vocab(self):
        torch = self.torch
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_()
        tokens = torch.tensor([[0, 1, 2, 3]])
        actual = response_log_probs_from_logits(logits, tokens, 2, 4)
        expected = torch.stack([logits[0, 1, :4].log_softmax(-1)[2], logits[0, 2, :4].log_softmax(-1)[3]])
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, :, 4:].abs().sum().item(), 0)
        single = response_log_probs_from_logits(logits[:, :2], tokens[:, :2], 1, 4)
        torch.testing.assert_close(single, logits[0, 0, :4].log_softmax(-1)[1:2])

    def test_both_advantage_signs_and_clip_boundaries(self):
        torch = self.torch
        # (advantage, ratio, expected loss, derivative wrt new logprob)
        for advantage, ratio, expected, gradient in ((1, 1.5, -1.2, 0), (1, .5, -.5, -.5),
                                                     (-1, .5, .8, 0), (-1, 1.5, 1.5, 1.5),
                                                     (1, 1, -1, -1), (-1, 1, 1, 1)):
            value = torch.tensor([math.log(ratio)], requires_grad=True)
            loss = clipped_policy_loss(value, torch.zeros(1), advantage, .2)
            loss.backward()
            self.assertAlmostEqual(loss.item(), expected, places=6)
            self.assertAlmostEqual(value.grad.item(), gradient, places=6)

    def test_parameter_update_and_zero_signal_preserves_adam_momentum(self):
        actor = self.actor()
        initial = actor.model.table.detach().clone()
        metrics = actor.update(self.samples(actor), [1, -1], .2, 2)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertFalse(self.torch.equal(initial, actor.model.table))
        changed = actor.model.table.detach().clone()
        with patch.object(actor.optimizer, "step", wraps=actor.optimizer.step) as step:
            metrics = actor.update(self.samples(actor), [0, 0], .2, 2)
            step.assert_not_called()
        self.assertFalse(metrics["update_applied"])
        self.assertEqual(actor.policy_version, 1)
        self.assertTrue(self.torch.equal(changed, actor.model.table))

    def test_nonfinite_gradient_blocks_step_and_export(self):
        actor = self.actor()
        actor.model.table.register_hook(lambda grad: grad * float("nan"))
        initial = actor.model.table.detach().clone()
        with patch.object(actor.optimizer, "step", wraps=actor.optimizer.step) as step:
            with self.assertRaises(RuntimeError):
                actor.update(self.samples(actor), [1, -1], .2, 2)
            step.assert_not_called()
        self.assertTrue(self.torch.equal(initial, actor.model.table))
        self.assertFalse(actor.valid)
        with self.assertRaisesRegex(RuntimeError, "Cannot publish"):
            actor.export_hf(None)

    def test_sequence_mean_gradient_accumulation(self):
        torch = self.torch
        actor = self.actor()
        actor.optimizer = torch.optim.SGD(actor.model.parameters(), lr=.1)
        reference = actor.model.table.detach().clone().requires_grad_()
        # Different lengths: each response has equal weight, each token 1/length within it.
        one = reference[0].log_softmax(-1)[1]
        two = (reference[0].log_softmax(-1)[2] + reference[2].log_softmax(-1)[1]) / 2
        loss = (-.25 * one + .25 * two) / 2
        loss.backward()
        actor.update(self.samples(actor, ([1], [2, 1])), [.25, -.25], .2, 2)
        torch.testing.assert_close(actor.model.table, reference.detach() - .1 * reference.grad)

    def test_nonfinite_loss_inputs_and_overflow_ratio_fail(self):
        torch = self.torch
        for new, old, advantage in ((float("nan"), 0, 1), (0, float("inf"), 1), (0, 0, float("nan")), (1000, 0, 1)):
            with self.assertRaises(ValueError):
                clipped_policy_loss(torch.tensor([new]), torch.tensor([old]), advantage, .2)


if __name__ == "__main__":
    unittest.main()
