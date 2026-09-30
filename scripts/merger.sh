#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/server_common.sh"

# Set this to the final checkpoint directory from your completed training run.
CKPT="${CKPT:-checkpoints/wmem-qwen2.5-7b/global_step_xxx}"
BASE="${BASE:-$MODEL_PATH}"
TARGET="${TARGET:-$CKPT/huggingface}"
require_file "$BASE/config.json"
if [[ ! -d "$CKPT/actor" ]]; then
    printf 'Missing actor checkpoint: %s/actor. Set CKPT to your final checkpoint.\n' "$CKPT" >&2
    exit 1
fi

python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "$BASE" \
    --local_dir "$CKPT/actor" \
    --target_dir "$TARGET"

# Export all tokenizer assets, including merges and special-token metadata.
python - "$BASE" "$TARGET" <<'PY'
import sys
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True).save_pretrained(sys.argv[2])
PY
printf 'Converted checkpoint: %s\n' "$TARGET"
