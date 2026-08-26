#!/usr/bin/env bash
#SBATCH --job-name=tx-repl-live
#SBATCH --partition=rtx-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

exec "${py}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle --train-get-image --partial-history \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --concurrent-expert-replay --sft-format tool_call --no-forced-json \
  --manifest training/bc_task_vlm/eval_manifests/transactional_replacement_regression_v1.json \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v4_transactional \
  --output-dir training/bc_task_vlm/eval_runs/transactional_replacement_regression_v1 \
  --success-criterion fsm --step-budget-factor 2.0 --gl-backend egl
