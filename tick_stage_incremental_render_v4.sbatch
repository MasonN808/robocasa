#!/usr/bin/env bash
#SBATCH --job-name=tick-stage-render-v4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python scripts/stage_incremental_render_dataset.py \
  --selection-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional \
  --processed-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional_image \
  --prior-render-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 \
  --output-render-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v4_transactional \
  --affected-input-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional_image_affected
