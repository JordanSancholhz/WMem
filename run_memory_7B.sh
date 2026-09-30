#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/server_common.sh"

export WANDB_MODE=offline
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/memcoe-ray-${USER:-user}-$$}"
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"

# All eight H200s train. The small frozen judge shares GPU 7 with a training rank.
NNODES=1
IFS=',' read -ra train_devices <<< "$CUDA_VISIBLE_DEVICES"
NGPUS_PER_NODE=${#train_devices[@]}
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_TOKEN_BUDGET="${PPO_TOKEN_BUDGET:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
if (( NGPUS_PER_NODE % ROLLOUT_TP_SIZE != 0 || (TRAIN_BATCH_SIZE * 2) % NGPUS_PER_NODE != 0 )); then
    echo 'GPU count must divide batch_size * rollout_n and be divisible by rollout TP size.' >&2
    exit 1
fi
printf 'Training GPUs: %s; global batch: %s; rollout TP: %s; samples per question: 2\n' \
    "$CUDA_VISIBLE_DEVICES" "$TRAIN_BATCH_SIZE" "$ROLLOUT_TP_SIZE"

TRAIN_PATH="${TRAIN_PATH:-${DATASET_ROOT}/RAG_top10_32k_train.parquet}"
FUTURE_PREDICTION_ENABLE="${FUTURE_PREDICTION_ENABLE:-True}"
FUTURE_PREDICTION_MODE="${FUTURE_PREDICTION_MODE:-known_state}"
PREDICTION_ALIGNMENT_ENABLE="${PREDICTION_ALIGNMENT_ENABLE:-True}"
PREDICTION_ALIGNMENT_STRENGTH="${PREDICTION_ALIGNMENT_STRENGTH:-0.1}"
LOCAL_CREDIT_ENABLE="${LOCAL_CREDIT_ENABLE:-False}"
LOCAL_CREDIT_COEFFICIENT="${LOCAL_CREDIT_COEFFICIENT:-0.05}"
LOCAL_CREDIT_STATE_STRENGTH="${LOCAL_CREDIT_STATE_STRENGTH:-0.1}"
WORLD_REWARD_ENABLE="${WORLD_REWARD_ENABLE:-True}"
WORLD_REWARD_COEFFICIENT="${WORLD_REWARD_COEFFICIENT:-0.5}"
WORLD_REWARD_STATE_STRENGTH="${WORLD_REWARD_STATE_STRENGTH:-0.1}"
DAMAGE_REWARD_ENABLE="${DAMAGE_REWARD_ENABLE:-True}"
DAMAGE_REWARD_COEFFICIENT="${DAMAGE_REWARD_COEFFICIENT:-0.01}"
DAMAGE_REWARD_MAX_UNITS="${DAMAGE_REWARD_MAX_UNITS:-2}"
DAMAGE_REWARD_SOURCE_CHARS="${DAMAGE_REWARD_SOURCE_CHARS:-1800}"
PREDICTION_CREDIT_ALPHA="${PREDICTION_CREDIT_ALPHA:-0.05}"
PREDICTION_CREDIT_DIRECTION="${PREDICTION_CREDIT_DIRECTION:-surprise}"
if [[ "${PREDICTION_ALIGNMENT_ENABLE,,}" == "true" || "${LOCAL_CREDIT_ENABLE,,}" == "true" || "${WORLD_REWARD_ENABLE,,}" == "true" ]]; then
    PREDICTION_CREDIT_ENABLE="${PREDICTION_CREDIT_ENABLE:-False}"
elif [[ "$FUTURE_PREDICTION_MODE" == "known_state" ]]; then
    PREDICTION_CREDIT_ENABLE="${PREDICTION_CREDIT_ENABLE:-$FUTURE_PREDICTION_ENABLE}"
else
    PREDICTION_CREDIT_ENABLE="${PREDICTION_CREDIT_ENABLE:-False}"
fi
if [[ "${DAMAGE_REWARD_ENABLE,,}" == "true" ]]; then
    if [[ "${LOCAL_CREDIT_ENABLE,,}" == "true" || "${PREDICTION_CREDIT_ENABLE,,}" == "true" ]]; then
        echo 'Evidence damage reward cannot be stacked with local/legacy advantage credit.' >&2
        exit 1
    fi
    # New default output: do not auto-resume/overwrite an existing Method11 run.
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_m11_damage_l${DAMAGE_REWARD_COEFFICIENT}_wm${WORLD_REWARD_ENABLE}_pred${FUTURE_PREDICTION_ENABLE}}"
fi
if [[ "${WORLD_REWARD_ENABLE,,}" == "true" ]]; then
    if [[ "${LOCAL_CREDIT_ENABLE,,}" == "true" || "${PREDICTION_CREDIT_ENABLE,,}" == "true" ||
          "${FUTURE_PREDICTION_ENABLE,,}" != "true" || "$FUTURE_PREDICTION_MODE" != "known_state" ]]; then
        echo 'World reward requires known_state prediction and both local/legacy advantage credit disabled.' >&2
        exit 1
    fi
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_method8_worldreward_g${WORLD_REWARD_COEFFICIENT}_eta${WORLD_REWARD_STATE_STRENGTH}}"
fi
if [[ "${LOCAL_CREDIT_ENABLE,,}" == "true" ]]; then
    if [[ "${PREDICTION_CREDIT_ENABLE,,}" == "true" || "${FUTURE_PREDICTION_ENABLE,,}" != "true" ||
          "$FUTURE_PREDICTION_MODE" != "known_state" ]]; then
        echo 'Local credit requires known_state prediction and legacy credit weighting disabled.' >&2
        exit 1
    fi
    # Separate from Method8/9/10 outputs so auto-resume cannot load those runs.
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_method8_local_l${LOCAL_CREDIT_COEFFICIENT}_eta${LOCAL_CREDIT_STATE_STRENGTH}}"
fi
if [[ "${PREDICTION_ALIGNMENT_ENABLE,,}" == "true" ]]; then
    if [[ "${PREDICTION_CREDIT_ENABLE,,}" == "true" || "${FUTURE_PREDICTION_ENABLE,,}" != "true" ||
          "$FUTURE_PREDICTION_MODE" != "known_state" ]]; then
        echo 'Gradient alignment requires known_state prediction and credit weighting disabled.' >&2
        exit 1
    fi
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_method5_alignment}"
elif [[ "${PREDICTION_CREDIT_ENABLE,,}" == "true" ]]; then
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_method5_${PREDICTION_CREDIT_DIRECTION}}"
else
    EXP="${EXP:-memory_agent/7B_8gpu_bs8_method5_decay}"
fi
printf 'Prediction credit: enabled=%s; alpha=%s; direction=%s; output=%s\n' \
    "$PREDICTION_CREDIT_ENABLE" "$PREDICTION_CREDIT_ALPHA" "$PREDICTION_CREDIT_DIRECTION" "$EXP"
printf 'Gradient alignment: enabled=%s; strength=%s; base=Method5 scheduled coefficient\n' \
    "$PREDICTION_ALIGNMENT_ENABLE" "$PREDICTION_ALIGNMENT_STRENGTH"
printf 'Local advantage correction (separate experiment): enabled=%s; lambda=%s; eta=%s\n' \
    "$LOCAL_CREDIT_ENABLE" "$LOCAL_CREDIT_COEFFICIENT" "$LOCAL_CREDIT_STATE_STRENGTH"
printf 'World guideline reward: enabled=%s; gamma=%s; eta=%s; normalized guideline weighting BEFORE GRPO\n' \
    "$WORLD_REWARD_ENABLE" "$WORLD_REWARD_COEFFICIENT" "$WORLD_REWARD_STATE_STRENGTH"
printf 'Evidence damage: enabled=%s; lambda=%s; units/update=%s; prior-source chars=%s; output=%s\n' \
    "$DAMAGE_REWARD_ENABLE" "$DAMAGE_REWARD_COEFFICIENT" "$DAMAGE_REWARD_MAX_UNITS" "$DAMAGE_REWARD_SOURCE_CHARS" "$EXP"
# Method2 supervision with Method4 time decay ONLY (no sparse-label scaling).
if [[ "$FUTURE_PREDICTION_MODE" == "known_state" ]]; then
    PREDICTION_SCHEDULE="${PREDICTION_SCHEDULE:-cosine_decay}"
else
    PREDICTION_SCHEDULE="${PREDICTION_SCHEDULE:-constant}"
fi
PROJ_DIR=${PROJ_ROOT}/${EXP}
require_file "$MODEL_PATH/config.json"
require_file "$TRAIN_PATH"
python scripts/check_model_service.py --judge
GUIDELINE_ARGS=()
if [[ -n "${GUIDELINE_PATH:-}" ]]; then
    require_file "$GUIDELINE_PATH"
    GUIDELINE_ARGS+=("recurrent.memory.config.guideline_path='$GUIDELINE_PATH'")
fi
if [[ -z "${MAX_CHUNKS:-}" ]]; then
    MAX_CHUNKS="$(python - "$MODEL_PATH" "$TRAIN_PATH" <<'PY'
import math
import sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
max_tokens = 0
for block in pq.ParquetFile(sys.argv[2]).iter_batches(columns=["context"], batch_size=128):
    for context in block.column(0).to_pylist():
        max_tokens = max(max_tokens, len(tokenizer.encode(context, add_special_tokens=False)))
print(max(8, math.ceil(max_tokens / 512)))
PY
)"
fi

# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=4096
MAX_NEW_TOKEN=256

exec python -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=512 \
    recurrent.memory.config.max_chunks="$MAX_CHUNKS" \
    recurrent.memory.config.intermediate_reward_enable=True \
    recurrent.memory.config.intermediate_reward_weight=0.5 \
    "recurrent.memory.config.intermediate_reward_model='$SERVED_MODEL_NAME'" \
    "recurrent.memory.config.intermediate_reward_base_url='$VLLM_URL'" \
    recurrent.memory.config.intermediate_reward_api_key_env=MEMCOE_API_KEY \
    recurrent.memory.config.intermediate_reward_reasoning_effort=null \
    recurrent.memory.config.intermediate_reward_temperature=0.0 \
    recurrent.memory.config.intermediate_reward_timeout=120 \
    recurrent.memory.config.intermediate_reward_concurrency=16 \
    recurrent.memory.config.intermediate_reward_max_completion_tokens=512 \
    recurrent.memory.config.damage_reward_enable="$DAMAGE_REWARD_ENABLE" \
    recurrent.memory.config.damage_reward_coefficient="$DAMAGE_REWARD_COEFFICIENT" \
    recurrent.memory.config.damage_reward_max_units="$DAMAGE_REWARD_MAX_UNITS" \
    recurrent.memory.config.damage_reward_source_chars="$DAMAGE_REWARD_SOURCE_CHARS" \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=40 \
    actor_rollout_ref.rollout.n=2 \
    'trainer.logger=[console]' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    "data.train_files=['$TRAIN_PATH']" \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    "actor_rollout_ref.model.path='$MODEL_PATH'" \
    actor_rollout_ref.distributed_timeout_seconds=900 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.future_prediction.enabled="$FUTURE_PREDICTION_ENABLE" \
    actor_rollout_ref.actor.future_prediction.mode="$FUTURE_PREDICTION_MODE" \
    actor_rollout_ref.actor.future_prediction.coefficient_schedule="$PREDICTION_SCHEDULE" \
    actor_rollout_ref.actor.future_prediction.gradient_alignment.enabled="$PREDICTION_ALIGNMENT_ENABLE" \
    actor_rollout_ref.actor.future_prediction.gradient_alignment.strength="$PREDICTION_ALIGNMENT_STRENGTH" \
    actor_rollout_ref.actor.future_prediction.credit_weighting.enabled="$PREDICTION_CREDIT_ENABLE" \
    actor_rollout_ref.actor.future_prediction.credit_weighting.coefficient="$PREDICTION_CREDIT_ALPHA" \
    actor_rollout_ref.actor.future_prediction.credit_weighting.direction="$PREDICTION_CREDIT_DIRECTION" \
    actor_rollout_ref.actor.future_prediction.local_credit.enabled="$LOCAL_CREDIT_ENABLE" \
    actor_rollout_ref.actor.future_prediction.local_credit.coefficient="$LOCAL_CREDIT_COEFFICIENT" \
    actor_rollout_ref.actor.future_prediction.local_credit.state_strength="$LOCAL_CREDIT_STATE_STRENGTH" \
    actor_rollout_ref.actor.future_prediction.world_reward.enabled="$WORLD_REWARD_ENABLE" \
    actor_rollout_ref.actor.future_prediction.world_reward.coefficient="$WORLD_REWARD_COEFFICIENT" \
    actor_rollout_ref.actor.future_prediction.world_reward.state_strength="$WORLD_REWARD_STATE_STRENGTH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_TOKEN_BUDGET" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_TOKEN_BUDGET" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_TOKEN_BUDGET" \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=memcoe \
    "trainer.experiment_name='$EXP'" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    "trainer.default_local_dir='$PROJ_DIR'" \
    trainer.total_epochs=5 \
    trainer.resume_mode=auto \
    "${GUIDELINE_ARGS[@]}" \
    "$@"


# trainer.resume_mode=resume_path \
