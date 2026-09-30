#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/server_common.sh"
require_file "$MODEL_PATH/config.json"
# Keep this service small enough to co-reside with training on an H200.
# gpu_memory_utilization is a vLLM allocation target, not a hard process limit.
export CUDA_VISIBLE_DEVICES="${SERVICE_GPUS:-7}"
IFS=',' read -ra service_devices <<< "$CUDA_VISIBLE_DEVICES"
TP_SIZE="${SERVICE_TP_SIZE:-${#service_devices[@]}}"
# MGI feedback contains two full memory trajectories, unlike a single rollout
# turn. Reserve enough context for both trajectories plus the completion.
MAX_MODEL_LEN="${SERVICE_MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${SERVICE_MAX_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
printf 'Serving %s as %s on GPUs %s; context=%s; batched tokens=%s\n' \
    "$MODEL_PATH" "$SERVED_MODEL_NAME" "$CUDA_VISIBLE_DEVICES" "$MAX_MODEL_LEN" "$MAX_BATCHED_TOKENS"
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "${SERVICE_HOST:-127.0.0.1}" \
    --port "${SERVICE_PORT:-6025}" \
    --api-key "$MEMCOE_API_KEY" \
    --tensor-parallel-size "$TP_SIZE" \
    --distributed-executor-backend mp \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "${SERVICE_MAX_NUM_SEQS:-16}" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "${SERVICE_GPU_MEMORY_UTILIZATION:-0.20}" \
    --enforce-eager \
    "$@"
