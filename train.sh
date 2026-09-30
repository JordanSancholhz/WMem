#!/usr/bin/env bash
# Public entry point for the final Method13 configuration.
set -euo pipefail
export EXP="${EXP:-checkpoints/wmem-qwen2.5-7b}"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_memory_7B.sh" "$@"
