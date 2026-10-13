"""One-GPU actor acceptance: HF logprobs, controlled GRPO update and parameter delta."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    actor = None
    report = {"status": "failed"}
    try:
        from .environment import preflight
        report["environment"] = preflight(args.model_path, "train", 256)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .actor import MegatronActor, response_log_probs_from_logits
        from .grpo import group_advantages
        from .reward import Example, ExactIntegerReward
        from .rollout import Sample, prompt_token_ids

        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        example = Example("What is 2 + 3? Reply with only the integer.", 5)
        prompt = prompt_token_ids(tokenizer, example.prompt)
        responses = [tokenizer.encode(text, add_special_tokens=False) for text in ("5", "6")]
        # CPU reference avoids allocating a second model on the training GPU.
        hf = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float32, attn_implementation="eager").eval()
        references = []
        with torch.no_grad():
            for response in responses:
                tokens = torch.tensor([prompt + response])
                references.append(response_log_probs_from_logits(hf(tokens).logits, tokens, len(prompt), hf.config.vocab_size))
        del hf
        actor = MegatronActor(args.model_path, learning_rate=1e-6)
        actor.model.eval()
        with torch.no_grad():
            before = [actor.response_log_probs(prompt, response).cpu() for response in responses]
        errors = []
        for actual, expected in zip(before, references):
            torch.testing.assert_close(actual, expected, atol=3e-3, rtol=1e-4)
            errors.append((actual - expected).abs().max().item())
        report["hf_logprob_max_abs_errors"] = errors
        report["logprob_tolerance"] = {"atol": 3e-3, "rtol": 1e-4}
        samples = [Sample(prompt, response, old.tolist(), text, f"controlled-{i}", "controlled", 0, "stop")
                   for i, (response, old, text) in enumerate(zip(responses, before, ("5", "6")))]
        rewards = [ExactIntegerReward()(example, sample.response_text) for sample in samples]
        advantages = group_advantages(rewards, 2)
        watched = actor.model.decoder.final_layernorm.weight
        parameter_before = watched.detach().clone()
        actor.model.train()
        report["update"] = actor.update(samples, advantages, clip_ratio=0.2, group_size=2)
        delta = (watched.detach() - parameter_before).abs().max().item()
        if not delta > 0 or not report["update"]["grad_norm"] > 0:
            raise AssertionError("Controlled update did not produce nonzero finite gradient and parameter change")
        with torch.no_grad():
            after = [actor.response_log_probs(prompt, response).cpu() for response in responses]
        if all(torch.equal(a, b) for a, b in zip(before, after)):
            raise AssertionError("Fixed continuation logprobs did not change")
        report.update(rewards=rewards, final_layernorm_max_abs_delta=delta,
                      before_logprobs=[x.tolist() for x in before], after_logprobs=[x.tolist() for x in after])
        actor.export_hf(args.output_dir / "updated_hf")
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if actor is not None:
                actor.close()
        except Exception as exc:
            report.update(status="failed", cleanup_error=str(exc))
            raise
        finally:
            path = args.output_dir / "actor-check.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "report": str(path)}), flush=True)


if __name__ == "__main__":
    main()
