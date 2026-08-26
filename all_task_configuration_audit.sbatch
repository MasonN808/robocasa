#!/usr/bin/env bash
#SBATCH --job-name=alltask-config-audit
#SBATCH --partition=rtx-batch
#SBATCH --array=0-52%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache

manifest=${MANIFEST:-training/bc_task_vlm/eval_manifests/all_task_configuration_audit_v1.json}
output_root=${OUTPUT_ROOT:-training/bc_task_vlm/eval_runs/all_task_configuration_audit_v1}
task=$(${PYTHON_BIN} -c "import json; d=json.load(open('${manifest}')); print(sorted(d['trajectory_ids_by_task'])[${SLURM_ARRAY_TASK_ID}])")

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle --train-get-image --partial-history \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --concurrent-expert-replay --sft-format tool_call --no-forced-json \
  --manifest "${manifest}" \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 \
  --output-dir "${output_root}/${task}" \
  --success-criterion fsm --step-budget-factor 2.0 --gl-backend egl \
  --tasks "${task}" --resume
