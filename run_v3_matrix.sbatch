#!/usr/bin/env bash
#SBATCH --job-name=v3-matrix
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# One script for the whole v3 matrix:
#   data source  : verbalized_sampling | structured_random
#   observability: centralized (--predict-acting-agent) | partial (--partial-history)
#   reasoning    : off | on (--train-reasoning)
#
# 4 GPUs rather than 8: whole-node 8-GPU requests could not backfill on
# b200-batch (656 pending / 79 running, every node mix|alloc; SLURM projected
# a ~7h wait). Effective batch is held constant at 64 by doubling grad-accum
# (4 procs x 8 per-device x 2 accum == 8 x 8 x 1).
#
# Partial-observability uses --partial-history WITHOUT --predict-acting-agent:
# dataset.py forbids partial_history+predict_agent ("the caller is the actor"
# -- there is no agent to predict when each example is one agent's own view),
# but explicitly supports partial_history+train_get_image.
#
# Select a cell with --export, e.g.:
#   sbatch --export=ALL,SOURCE=structured_random,OBS=partial,REASONING=1 run_v3_matrix.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${SOURCE:?Set SOURCE to verbalized_sampling or structured_random}"
: "${OBS:?Set OBS to centralized or partial}"
REASONING=${REASONING:-0}

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

case "${SOURCE}" in
  verbalized_sampling)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
    src_tag=vs
    ;;
  structured_random)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_train_21
    src_tag=sr
    ;;
  *) echo "Unknown SOURCE: ${SOURCE}" >&2; exit 1 ;;
esac

obs_args=()
case "${OBS}" in
  centralized)
    obs_args=(--predict-acting-agent --train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_args=(
      --partial-history
      --train-get-image
      --partial-step-index-mode local
      --partial-observation-mode consume-once
    )
    obs_tag=partial
    ;;
  *) echo "Unknown OBS: ${OBS}" >&2; exit 1 ;;
esac

reason_args=()
reason_tag=noreason
if [[ "${REASONING}" == "1" ]]; then
  reason_args=(--train-reasoning)
  reason_tag=reason
fi

# structured_random's batch composition (high share of no-image
# communicate/give_space steps) crashes DDP with
# ddp_find_unused_parameters=True + gradient checkpointing
# ("Expected to mark a variable ready only once"). Disabling checkpointing is
# the fix that actually worked; non-reentrant checkpointing was not enough.
ckpt_args=()
if [[ "${SOURCE}" == "structured_random" ]]; then
  ckpt_args=(--no-gradient-checkpointing)
fi

run_name="qwen3vl-8b-v3-${src_tag}-${obs_tag}-${reason_tag}"
export OUTPUT_DIR="training/bc_task_vlm/runs/${run_name}"
export WANDB_RUN_NAME="${run_name}"

# --- 4 GPUs, effective batch 64 (== the 8-GPU config's 8x8x1) ---
export NUM_PROCESSES=4
export PER_DEVICE_BATCH_SIZE=8
export GRAD_ACCUM=2
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

export NUM_EPOCHS=3
export EVAL_STEPS=2000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2
export EVAL_MAX_SAMPLES=512
export EVAL_GENERATION_MAX_SAMPLES=0

echo "=== ${run_name} ==="
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  obs=${obs_args[*]}"
echo "  reasoning=${reason_args[*]:-none}  ckpt=${ckpt_args[*]:-default}"

source training/scripts/local/_launch_args.sh
launch_args+=(
  "${obs_args[@]}"
  "${reason_args[@]}"
  "${ckpt_args[@]}"
  --ddp-timeout-seconds 1800
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${run_name}"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
