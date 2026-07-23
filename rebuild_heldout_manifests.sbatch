#!/usr/bin/env bash
#SBATCH --job-name=rebuild-heldout-manifests
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Rebuilds the heldout_trajectories/heldout_tasks/pilot eval manifests now
# that the stale hf_stage_manifest.json (which was silently overwritten,
# not merged, by each held-out-task staging call and caused
# _trajectory_dirs_for_task to resolve to 0 trajectories for every task not
# in its last-written "sources" list) has been moved aside, so
# _trajectory_dirs_for_task falls back to plain directory listing.
#   sbatch rebuild_heldout_manifests.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

"${PYTHON_BIN}" -m training.bc_task_vlm.build_eval_manifest \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset \
  --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
  --output-dir training/bc_task_vlm/eval_manifests/v1_5_reasoning

echo "Done."
