#!/usr/bin/env bash
#SBATCH --job-name=atomic-handover
#SBATCH --partition=rtx-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_dir=${OUTPUT_DIR:-training/bc_task_vlm/eval_runs/atomic_handover_integration_v1}

exec "${python_bin}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle --train-get-image --partial-history \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --concurrent-expert-replay --sft-format tool_call --no-forced-json \
  --manifest training/bc_task_vlm/eval_manifests/atomic_handover_integration_v1.json \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 \
  --output-dir "${output_dir}" \
  --success-criterion fsm --step-budget-factor 2.0 --gl-backend egl \
  --tasks arrange_bread_bowl
