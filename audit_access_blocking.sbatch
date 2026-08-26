#!/usr/bin/env bash
#SBATCH --job-name=access-audit
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
DATASET_ROOT=${DATASET_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3}
OUTPUT_DIR=${OUTPUT_DIR:-training/bc_task_vlm/eval_runs/access_blocking_audit_v1}
TASKS=${TASKS:-}
args=(--dataset-root "$DATASET_ROOT" --output-dir "$OUTPUT_DIR" --gl-backend egl)
if [[ -n "$TASKS" ]]; then args+=(--tasks "$TASKS"); fi
"$PY" -m training.bc_task_vlm.audit_access_blocking "${args[@]}"
if [[ "${EXPECT_DETECTED:-0}" == "1" ]]; then
  "$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["num_detected"] > 0, "smoke detected no blocking cases"' "$OUTPUT_DIR/audit.json"
fi
