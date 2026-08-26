#!/usr/bin/env bash
#SBATCH --job-name=materialize-nrs-s30-t43
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.materialize_scale_artifacts \
  --master /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_master_train47_noindex_promptv9 \
  --selection training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/selection_30.json \
  --output /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_noreason_scale30_train43_noindex_promptv9
