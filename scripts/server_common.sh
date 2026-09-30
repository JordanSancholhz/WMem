#!/usr/bin/env bash
# Source from any working directory. All model paths are local filesystem paths.
PROJ_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PATH="${MODEL_PATH:-$PROJ_ROOT/models/Qwen2.5-7B-Instruct}"
export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$PROJ_ROOT/sentence-transformers/all-MiniLM-L6-v2}"
export DATASET_ROOT="${DATASET_ROOT:-$PROJ_ROOT/MGI_module/datasets/PersonaMem}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct}"
export VLLM_URL="${VLLM_URL:-http://127.0.0.1:6025/v1}"
export MEMCOE_API_KEY="${MEMCOE_API_KEY:-memcoe-local}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'Missing required file: %s\n' "$1" >&2
        exit 1
    fi
}
