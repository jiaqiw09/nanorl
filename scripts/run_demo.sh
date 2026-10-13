#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/Qwen2.5-0.5B-Instruct [trainer arguments...]" >&2
  exit 2
fi
# Resolve the caller's paths before changing directory.
MODEL_PATH="$(cd "$1" && pwd)"
shift
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
cd "$PROJECT_DIR"
exec "${PYTHON:-python3}" -m nanorl.launch \
  --model-path "$MODEL_PATH" \
  --data-path "$PROJECT_DIR/examples/math.jsonl" \
  --output-dir "$RUN_DIR/weights" \
  --sglang-url "http://127.0.0.1:${PORT:-30000}" \
  "$@"
