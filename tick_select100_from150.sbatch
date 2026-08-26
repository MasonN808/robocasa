#!/usr/bin/env bash
#SBATCH --job-name=tick-select100
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python scripts/select_generated_trajectory_subset.py --input-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_partition_none_temp06_v1 --output-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v1 --per-task 100 --expected-tasks 53 --seed 42
