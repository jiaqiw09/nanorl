#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ @sha256:[a-f0-9]{64}$ ]]; then
  echo "Usage: $0 repository@sha256:<64 hex digits> /path/to/local/model [trainer arguments...]" >&2
  exit 2
fi
IMAGE="$1"
MODEL_PATH="$(cd "$2" && pwd)"
shift 2
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The image must already contain the complete SGLang/Bridge/CUDA stack.
# Never install floating dependencies while starting a training run.
exec docker run --rm --init --gpus all --ipc=host \
  --mount "type=bind,src=$PROJECT_DIR,dst=/workspace/nanorl" \
  --mount "type=bind,src=$MODEL_PATH,dst=/models/qwen,readonly" \
  --workdir /workspace/nanorl \
  --env "NANORL_IMAGE=$IMAGE" \
  --env "TRAIN_GPU=${TRAIN_GPU:-0}" \
  --env "ROLLOUT_GPU=${ROLLOUT_GPU:-1}" \
  --env "PORT=${PORT:-30000}" \
  --env "MASTER_PORT=${MASTER_PORT:-29501}" \
  --env "STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-300}" \
  "$IMAGE" bash scripts/run_demo.sh /models/qwen "$@"
