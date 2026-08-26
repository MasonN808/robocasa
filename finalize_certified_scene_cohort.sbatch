#!/usr/bin/env bash
#SBATCH --job-name=scene-finalize
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
AUDIT_DIR=training/bc_task_vlm/eval_runs/certified_scene_compatibility_v1
WORKSPACE_AUDIT_DIR=training/bc_task_vlm/eval_runs/task_scene_shared_workspace_v1
INITIALIZATION_CACHE=training/bc_task_vlm/eval_manifests/certified_scene_initialization_v1.json
CACHE=training/bc_task_vlm/eval_manifests/certified_scene_compatibility_v1.json
TARGET_DIR=training/bc_task_vlm/eval_manifests/fixed_live_sim_state_grounded_v3_certified_scenes
TARGETS=${TARGET_DIR}/configuration_targets.json

"${PYTHON_BIN}" -m training.bc_task_vlm.build_scene_compatibility_cache \
  --input-dir "${AUDIT_DIR}" \
  --output "${INITIALIZATION_CACHE}" \
  --require-scene-count 60 \
  --allow-task-errors
"${PYTHON_BIN}" -m training.bc_task_vlm.merge_shared_workspace_compatibility \
  --initialization-cache "${INITIALIZATION_CACHE}" \
  --workspace-audit-dir "${WORKSPACE_AUDIT_DIR}" \
  --output "${CACHE}" \
  --require-workspace-scene-count 75
"${PYTHON_BIN}" -m training.bc_task_vlm.build_fixed_cohort_targets \
  --initial-config-manifest training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150_canonical_workspace.json \
  --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
  --output "${TARGETS}"
"${PYTHON_BIN}" -m training.bc_task_vlm.verify_scene_cohort_contract \
  --cache "${CACHE}" \
  --targets "${TARGETS}" \
  --output "${TARGET_DIR}/scene_contract_verification.json"
