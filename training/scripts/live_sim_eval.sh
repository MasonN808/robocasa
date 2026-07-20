#!/usr/bin/env bash
#SBATCH --job-name=live-sim-eval
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail

# Override these with sbatch --export=ALL,NAME=value.  EXTRA_ARGS may contain
# filters such as "--tasks garnish_cake --max-trajectories 2".
BACKEND=${BACKEND:-oracle}
SPLIT=${SPLIT:-heldout_tasks}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-}
RUN_NAME=${RUN_NAME:-${BACKEND}__${SPLIT}}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p slurm_logs
export HF_HOME=${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

args=(
  --backend "$BACKEND"
  --manifest "training/bc_task_vlm/eval_manifests/exp52/eval_manifest_${SPLIT}.json"
  --dataset-root training/bc_task_vlm/eval_data_subset
  --output-dir "training/bc_task_vlm/eval_runs/live_sim_${RUN_NAME}"
  --gl-backend egl
  --resume
)
if [[ "$BACKEND" == hf ]]; then
  args+=(--model-name-or-path "$MODEL")
elif [[ "$BACKEND" == gemini ]]; then
  args+=(--model "$MODEL")
fi
if [[ -n "$ADAPTER" ]]; then
  args+=(--adapter-path "$ADAPTER")
fi
# Intentional word splitting lets callers pass several ordinary CLI flags.
# shellcheck disable=SC2206
extra=( $EXTRA_ARGS )
args+=("${extra[@]}")

.venv/bin/python -m training.bc_task_vlm.live_sim_eval "${args[@]}"
