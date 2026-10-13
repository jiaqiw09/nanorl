"""CPU numerical tests; the actual Bridge acceptance entry is nanorl.roundtrip."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from nanorl.conversion import HFCheckpoint, compare_checkpoints, compare_tensor, expected_shapes, hf_views, verify_megatron


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "CPU torch and safetensors required for numerical conversion tests")
class ConversionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.config = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 8,
                       "intermediate_size": 12, "num_hidden_layers": 1, "num_attention_heads": 4,
                       "num_key_value_heads": 2, "vocab_size": 10, "max_position_embeddings": 32,
                       "tie_word_embeddings": True}

    def checkpoint(self, path, *, omit=None, mutate=None):
        from safetensors.torch import save_file
        path.mkdir()
        (path / "config.json").write_text(json.dumps(self.config))
        weights = {name: self.torch.arange(self.torch.tensor(shape).prod().item(), dtype=self.torch.float32).reshape(shape) + i * 1000
                   for i, (name, shape) in enumerate(expected_shapes(self.config).items()) if name != "lm_head.weight" and name != omit}
        if mutate:
            mutate(weights)
        save_file(weights, str(path / "model.safetensors"))
        return HFCheckpoint(path)

    def test_gqa_weight_and_bias_group_order(self):
        for kind, tensor in (("weight", self.torch.arange(16 * 8).reshape(16, 8)),
                             ("bias", self.torch.arange(16))):
            views = hf_views(f"decoder.layers.0.self_attention.linear_qkv.{kind}", tensor, self.config)
            for label, rows in (("q", [0, 1, 2, 3, 8, 9, 10, 11]), ("k", [4, 5, 12, 13]), ("v", [6, 7, 14, 15])):
                self.assertTrue(self.torch.equal(views[f"model.layers.0.self_attn.{label}_proj.{kind}"], tensor[rows]))

    def test_gate_up_order_and_unknown_mapping(self):
        tensor = self.torch.arange(24 * 8).reshape(24, 8)
        views = hf_views("decoder.layers.0.mlp.linear_fc1.weight", tensor, self.config)
        self.assertTrue(self.torch.equal(views["model.layers.0.mlp.gate_proj.weight"], tensor[:12]))
        self.assertTrue(self.torch.equal(views["model.layers.0.mlp.up_proj.weight"], tensor[12:]))
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            hf_views("decoder.layers.0.unmapped.weight", tensor, self.config)

    def test_middle_state_detects_swapped_kv_and_unshared_head(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            source = self.checkpoint(Path(directory) / "source")
            params = {"embedding.word_embeddings.weight": source.tensor("model.embed_tokens.weight"),
                      "decoder.final_layernorm.weight": source.tensor("model.norm.weight")}
            layer = "model.layers.0."
            for mc, hf in {"self_attention.linear_proj.weight": "self_attn.o_proj.weight",
                           "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
                           "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
                           "mlp.linear_fc2.weight": "mlp.down_proj.weight"}.items():
                params["decoder.layers.0." + mc] = source.tensor(layer + hf)
            for kind in ("weight", "bias"):
                q, k, v = [source.tensor(layer + f"self_attn.{label}_proj.{kind}") for label in ("q", "k", "v")]
                # Explicit two GQA groups; each contains four Q rows, two K rows, two V rows.
                params["decoder.layers.0.self_attention.linear_qkv." + kind] = self.torch.cat(
                    [q[:4], k[:2], v[:2], q[4:], k[2:], v[2:]])
            params["decoder.layers.0.mlp.linear_fc1.weight"] = self.torch.cat(
                [source.tensor(layer + "mlp.gate_proj.weight"), source.tensor(layer + "mlp.up_proj.weight")])
            embed = params["embedding.word_embeddings.weight"]
            model = SimpleNamespace(share_embeddings_and_output_weights=True, output_layer=SimpleNamespace(weight=None),
                                    embedding=SimpleNamespace(word_embeddings=SimpleNamespace(weight=embed)),
                                    named_parameters=lambda: params.items())
            verify_megatron(model, source)
            key = "decoder.layers.0.self_attention.linear_qkv.weight"
            original = params[key].clone()
            params[key][4:6], params[key][6:8] = original[6:8], original[4:6]
            with self.assertRaisesRegex(ValueError, "max_abs_error"):
                verify_megatron(model, source)
            params[key] = original
            model.output_layer.weight = embed.clone()
            with self.assertRaisesRegex(ValueError, "not actually shared"):
                verify_megatron(model, source)

    def test_roundtrip_and_missing_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.checkpoint(root / "a"), self.checkpoint(root / "b")
            self.assertTrue(all(row["max_abs_error"] == 0 for row in compare_checkpoints(a, b)))
            with self.assertRaisesRegex(ValueError, "missing"):
                self.checkpoint(root / "missing", omit="model.layers.0.self_attn.q_proj.bias")

    def test_corrupt_values_shapes_and_tied_weights_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.checkpoint(root / "source")
            def wrong_q(weights):
                weights["model.layers.0.self_attn.q_proj.weight"][0, 0] += 1
            bad = self.checkpoint(root / "bad", mutate=wrong_q)
            with self.assertRaisesRegex(ValueError, "max_abs_error"):
                compare_checkpoints(source, bad)
            def wrong_shape(weights):
                weights["model.layers.0.mlp.gate_proj.weight"] = self.torch.zeros(1)
            with self.assertRaisesRegex(ValueError, "shape"):
                self.checkpoint(root / "shape", mutate=wrong_shape)
            def untie(weights):
                weights["lm_head.weight"] = weights["model.embed_tokens.weight"] + 1
            with self.assertRaisesRegex(ValueError, "Tied"):
                self.checkpoint(root / "untie", mutate=untie)

    def test_index_missing_shard_and_wrong_location_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            checkpoint = self.checkpoint(path)
            mapping = {name: "missing.safetensors" for name in checkpoint.locations}
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            with self.assertRaises(FileNotFoundError):
                HFCheckpoint(path)
            mapping = {name: "model.safetensors" for name in list(checkpoint.locations)[1:]}
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
            with self.assertRaisesRegex(ValueError, "index"):
                HFCheckpoint(path)

    def test_nonfinite_and_dtype_policy(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_tensor("nan", self.torch.tensor([float("nan")]), self.torch.ones(1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def bf16(weights):
                for name in weights:
                    weights[name] = weights[name].bfloat16()
            source = self.checkpoint(root / "source", mutate=bf16)
            def widened(weights):
                for name in weights:
                    weights[name] = weights[name].bfloat16().float()
            exported = self.checkpoint(root / "exported", mutate=widened)
            compare_checkpoints(source, exported)
            with self.assertRaisesRegex(ValueError, "dtype"):
                compare_checkpoints(exported, source)


if __name__ == "__main__":
    unittest.main()
