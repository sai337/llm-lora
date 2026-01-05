#!/usr/bin/env bash
set -euo pipefail

# vLLM CPU docs: you may need to build from source on x86. citeturn0search0turn0search12
# CLI args doc: vllm serve supports LoRA flags. citeturn0search4turn0search1turn0search5
#
# Usage:
#   bash scripts/serve_vllm.sh outputs/<run_id>/export
# or
#   bash scripts/serve_vllm.sh gpt2

MODEL_PATH="${1:-gpt2}"
API_KEY="${API_KEY:-local-dev-key}"
PORT="${PORT:-8000}"

echo "Starting vLLM server on :$PORT model=$MODEL_PATH"
# If your vLLM build supports --device cpu, you can add it. Some CPU builds auto-detect.
vllm serve "$MODEL_PATH" --dtype auto --port "$PORT" --api-key "$API_KEY"
