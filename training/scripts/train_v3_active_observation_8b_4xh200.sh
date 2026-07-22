#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v3-4xh200
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougol@umass.edu

# Four-H200 v3 active-observation LoRA SFT.
#
# Defaults:
#   4 ranks x per-device batch 4 x grad-accum 1 = global batch 16
#   FlashAttention 2, three epochs, capped loss eval, no generation eval
#
# Smoke test:
#   MAX_STEPS=20 WANDB_RUN_NAME=qwen3vl8b-v3-4xh200-smoke \
#     OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl8b-v3-4xh200-smoke \
#     sbatch --time=02:00:00 training/scripts/train_v3_active_observation_8b_4xh200.sh
#
# Opt into the separately tested length-grouped sampler with:
#   TRAIN_SAMPLING_STRATEGY=group_by_length sbatch ...

set -euo pipefail
mkdir -p slurm_logs

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
cd "${repo_root}"
export UV_CACHE_DIR=/work/hdd/bgjs/dbenhamougoldfajn/.uv_cache
export HF_HOME="${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export PYTHON_BIN="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
export DATA_ROOT="${DATA_ROOT:-/work/hdd/bgjs/dbenhamougoldfajn/robocasa_agentsft_subset}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/bc_task_vlm/runs/qwen3vl-8b-active-observation-4xh200}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export TRAIN_SAMPLING_STRATEGY="${TRAIN_SAMPLING_STRATEGY:-random}"
export EVAL_STEPS="${EVAL_STEPS:-2000}"
export SAVE_STEPS="${SAVE_STEPS:-2000}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-512}"
export EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3vl-8b-active-observation-4xh200}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"

[[ -d "${DATA_ROOT}" ]] || {
  echo "DATA_ROOT ${DATA_ROOT} missing - stage it before training" >&2
  exit 1
}

# The intended full v3 experiment uses all 47 selected train tasks with at
# least 30 trajectories each before applying the deterministic 10% holdout.
"${PYTHON_BIN}" - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
selection_path = Path(
    "data_analysis/analysis/held_out_task_selection/held_out_task_selection.json"
)
tasks = json.loads(selection_path.read_text())["train_tasks"]
counts = {
    task: len([path for path in (dataset_root / task).glob("traj_*") if path.is_dir()])
    for task in tasks
}
undersized = {task: count for task, count in counts.items() if count < 30}
if undersized:
    details = ", ".join(f"{task}={count}" for task, count in undersized.items())
    raise SystemExit(
        f"DATA_ROOT is not the 30-trajectories/task dataset; undersized tasks: {details}"
    )
print(f"Validated {len(tasks)} training tasks with at least 30 trajectories each.")
PY

# Defines the common Qwen3-VL 8B model, split, LoRA, precision, image, and
# optimizer arguments. Environment variables above intentionally override its
# local-box defaults.
source "${repo_root}/training/scripts/local/_launch_args.sh"

launch_args+=(
  --predict-acting-agent
  --train-get-image
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  launch_args+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  launch_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

global_batch=$((PER_DEVICE_BATCH_SIZE * GRAD_ACCUM * NUM_PROCESSES))
echo "Launching v3 active-observation SFT on ${NUM_PROCESSES} H200 GPUs"
echo "  per_device=${PER_DEVICE_BATCH_SIZE} grad_accum=${GRAD_ACCUM} global_batch=${global_batch}"
echo "  epochs=${NUM_EPOCHS} attention=${ATTN_IMPLEMENTATION} sampler=${TRAIN_SAMPLING_STRATEGY}"
echo "  dataset=${DATA_ROOT} output=${OUTPUT_DIR}"

exec "${repo_root}/training/scripts/launch_multigpu.sh" "${launch_args[@]}"
