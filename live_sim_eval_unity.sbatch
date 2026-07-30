#!/usr/bin/env bash
#SBATCH --job-name=live-sim-eval
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Unity port of training/scripts/live_sim_eval.sh (which is Delta-specific:
# --account=bgjs-delta-gpu, /work/hdd/... chdir, .venv/bin/python -- none of
# which exist on this cluster). Same config surface via --export=ALL,...:
# BACKEND, SPLIT, MODEL, ADAPTER, RUN_NAME, MANIFEST_DIR, DATASET_ROOT,
# EXTRA_ARGS. MANIFEST_DIR/DATASET_ROOT default to exp52 (v1's real eval
# data); override for structured_random.
#
# Follows LIVE_SIM_EVAL.md's recommended gate sequence -- run oracle, then a
# 2-trajectory model pilot, before a full split:
#   sbatch --export=ALL,BACKEND=oracle,SPLIT=heldout_tasks,\
#     MANIFEST_DIR=training/bc_task_vlm/eval_manifests/structured_random_30,\
#     DATASET_ROOT=/work/.../data/structured_random_eval_root,\
#     RUN_NAME=oracle_gate_sr,EXTRA_ARGS='--tasks beverage_organization --max-trajectories 1 --no-record-firsts' \
#     live_sim_eval_unity.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

BACKEND=${BACKEND:-hf}
SPLIT=${SPLIT:-heldout_tasks}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-}
MANIFEST_DIR=${MANIFEST_DIR:-training/bc_task_vlm/eval_manifests/exp52}
DATASET_ROOT=${DATASET_ROOT:-training/bc_task_vlm/eval_data_subset}
RUN_NAME=${RUN_NAME:-${BACKEND}__${SPLIT}}
EXTRA_ARGS=${EXTRA_ARGS:-}

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

args=(
  --backend "$BACKEND"
  --manifest "${MANIFEST_DIR}/eval_manifest_${SPLIT}.json"
  --dataset-root "$DATASET_ROOT"
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

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval "${args[@]}"
