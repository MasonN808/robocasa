#!/usr/bin/env bash
#SBATCH --job-name=materialize-tick150-scales
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
scales=(30 60 90 120 150)
scale="${scales[SLURM_ARRAY_TASK_ID]}"
cd "${repo}"
exec "${python_bin}" -m training.bc_task_vlm.materialize_scale_artifacts \
  --master /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_master_train47_noindex_promptv9 \
  --selection "training/bc_task_vlm/scale_experiments/tick150_v1/selection_${scale}.json" \
  --output "/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_${scale}_train47_noindex_promptv9"
