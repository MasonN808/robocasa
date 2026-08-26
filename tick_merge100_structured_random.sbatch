#!/usr/bin/env bash
#SBATCH --job-name=tick-merge100sr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
"${python_bin}" scripts/merge_rendered_trajectory_corpora.py \
  --base-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent \
  --supplement-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render70_structured_random_concurrent \
  --output-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_structured_random_concurrent \
  --expected-per-task 100
