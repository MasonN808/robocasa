#!/usr/bin/env bash
#SBATCH --job-name=materialize-rsn-t43-scale
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

: "${SCALE:?Set SCALE to 30, 60, 90, 120, or 150}"
case "${SCALE}" in
  30|60|90|120|150) ;;
  *) echo "Unsupported SCALE=${SCALE}" >&2; exit 2 ;;
esac

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.materialize_scale_artifacts \
  --master /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_reasoning_master_train47_noindex_promptv9 \
  --selection "training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/selection_${SCALE}.json" \
  --output "/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_reasoning_scale${SCALE}_train43_noindex_promptv9"
