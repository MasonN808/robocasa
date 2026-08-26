#!/usr/bin/env bash
#SBATCH --job-name=mat-t43-scales
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
scales=(60 90 120 150 60)
kinds=(noreason noreason noreason noreason reasoning)
idx=${SLURM_ARRAY_TASK_ID}
scale=${scales[idx]}
kind=${kinds[idx]}
master=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_master_train47_noindex_promptv9
if [[ "${kind}" == reasoning ]]; then
  master=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_reasoning_master_train47_noindex_promptv9
fi
output=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_${kind}_scale${scale}_train43_noindex_promptv9
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  -m training.bc_task_vlm.materialize_scale_artifacts \
  --master "${master}" \
  --selection "training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/selection_${scale}.json" \
  --output "${output}"
