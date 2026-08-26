#!/usr/bin/env bash
#SBATCH --job-name=excl-ws-replay
#SBATCH --partition=rtx-batch
#SBATCH --array=0-105%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache

manifest=training/bc_task_vlm/eval_manifests/exclusive_workspace_full_pool.json
task_index=$((SLURM_ARRAY_TASK_ID % 53))
arm_index=$((SLURM_ARRAY_TASK_ID / 53))
task=$(${PYTHON_BIN} -c "import json; d=json.load(open('${manifest}')); print(sorted(d['trajectory_ids_by_task'])[${task_index}])")
if [[ ${arm_index} -eq 0 ]]; then
  export ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES=0
  arm=baseline
else
  export ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES=1
  arm=reserved
fi

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle --train-get-image --partial-history \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --concurrent-expert-replay --sft-format tool_call --no-forced-json \
  --manifest "${manifest}" \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 \
  --output-dir "training/bc_task_vlm/eval_runs/exclusive_workspace_full_replay_${arm}/${task}" \
  --success-criterion fsm --step-budget-factor 2.0 --gl-backend egl \
  --tasks "${task}" --resume
