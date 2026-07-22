#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v3-observation-sft
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256g
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Fresh v3 LoRA: v2 agent prediction plus supervised, model-driven get_image.
#   sbatch training/scripts/train_v3_active_observation_8b_cluster.sh
set -euo pipefail
mkdir -p slurm_logs

export UV_CACHE_DIR=/work/hdd/bgjs/dbenhamougoldfajn/.uv_cache
export HF_HOME="${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

export PYTHON_BIN="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/.venv/bin/python"
export DATA_ROOT="${DATA_ROOT:-/work/hdd/bgjs/dbenhamougoldfajn/robocasa_agentsft_subset}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/bc_task_vlm/runs/qwen3vl-8b-active-observation}"
export PREDICT_ACTING_AGENT=1
export TRAIN_GET_IMAGE=1
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export EVAL_STEPS="${EVAL_STEPS:-2000}"
export SAVE_STEPS="${SAVE_STEPS:-2000}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-512}"
export EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:-0}"
export WANDB_MODE=online
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3vl-8b-active-observation}"

[ -d "${DATA_ROOT}" ] || { echo "DATA_ROOT ${DATA_ROOT} missing - stage first"; exit 1; }

# Refuse another silently undersized run. The intended v3 experiment requires
# 30 trajectories for every selected training task before the 10% validation
# holdout is applied.
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

exec bash training/scripts/local/train_qwen3vl_8b.sh
