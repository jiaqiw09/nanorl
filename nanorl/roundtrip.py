"""Real one-GPU HF -> Megatron -> HF acceptance test (no optimizer step)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .conversion import HFCheckpoint, compare_checkpoints, verify_megatron
from .environment import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "input": str(args.model_path.resolve()), "stages": {}}
    actor = None
    try:
        report["environment"] = preflight(args.model_path, "train", 256)
        from transformers import AutoTokenizer
        from .actor import MegatronActor

        actor = MegatronActor(args.model_path, learning_rate=1e-6)
        source = actor.conversion.source
        # Independent middle-state verification prevents paired inverse bugs cancelling out.
        report["stages"]["hf_to_megatron"] = verify_megatron(actor.model, source)
        exported_path = args.output_dir / "exported_hf"
        actor.export_hf(exported_path)
        report["stages"]["hf_roundtrip"] = compare_checkpoints(source, HFCheckpoint(exported_path))
        original = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
        exported = AutoTokenizer.from_pretrained(exported_path, local_files_only=True, trust_remote_code=False)
        for field in ("chat_template", "special_tokens_map", "model_max_length", "padding_side", "truncation_side"):
            if getattr(original, field) != getattr(exported, field):
                raise ValueError(f"Tokenizer changed: {field}")
        if original.get_vocab() != exported.get_vocab():
            raise ValueError("Exported tokenizer vocabulary changed")
        for text in ("1+1?", "你好，计算 12 + 7。", "\n-123 = 456"):
            messages = [{"role": "user", "content": text}]
            if original.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) != exported.apply_chat_template(messages, tokenize=True, add_generation_prompt=True):
                raise ValueError("Exported tokenizer encoding changed")
        report["stages"]["tokenizer"] = "passed"
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if actor is not None:
                actor.close()
        except Exception as exc:
            report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (args.output_dir / "roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "report": str(args.output_dir / "roundtrip.json")}), flush=True)


if __name__ == "__main__":
    main()
