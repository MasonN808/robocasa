#!/usr/bin/env bash
#SBATCH --job-name=stage-heldout-eval-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Stages the 5 fully-held-out tasks (garnish_cake, meat_skewer_assembly,
# hot_dog_setup, cluster_items_for_clearing, beverage_organization) from HF
# into the same dataset root as the 47 training tasks
# (robocasa_agentsft_subset), then rebuilds the fixed eval manifests
# (heldout_trajectories / heldout_tasks / pilot) needed to eval the v1.5
# reasoning checkpoint against v1's own heldout_tasks transfer split -- these
# manifests don't exist on this cluster yet (only training data was staged).
# CPU-only; no GPU needed.
#   sbatch stage_heldout_eval_data.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset
export SELECTION=data_analysis/analysis/held_out_task_selection/held_out_task_selection.json
export HF_PREFIX="DorianAtSchool/robocasa_20260430T030150Z_full_run_"
export TRAJ_PER_TASK=30
export MANIFEST_OUTPUT_DIR=training/bc_task_vlm/eval_manifests/v1_5_reasoning

mapfile -t held_out_tasks < <("${PYTHON_BIN}" -c \
  'import json; print("\n".join(json.load(open("'"${SELECTION}"'"))["held_out_tasks"]))')

echo "Staging ${#held_out_tasks[@]} held-out tasks -> ${DATASET_ROOT}"
for task in "${held_out_tasks[@]}"; do
  echo "=== ${task} ==="
  "${PYTHON_BIN}" training/scripts/stage_hf_sweep_datasets.py \
    --repo-id "${HF_PREFIX}${task}" \
    --output-root "${DATASET_ROOT}" \
    --max-total-episodes "${TRAJ_PER_TASK}" \
    --workers 8 \
    --resume
done

echo "Rebuilding eval manifests -> ${MANIFEST_OUTPUT_DIR}"
"${PYTHON_BIN}" -m training.bc_task_vlm.build_eval_manifest \
  --dataset-root "${DATASET_ROOT}" \
  --selection "${SELECTION}" \
  --output-dir "${MANIFEST_OUTPUT_DIR}"

echo "Done."
