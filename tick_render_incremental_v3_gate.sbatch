#!/usr/bin/env bash
#SBATCH --job-name=tick-render-v3-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
metadata="$(find "${root}" -mindepth 3 -maxdepth 3 -name trajectory_execution_metadata.json -type f | wc -l)"
plans="$(find "${root}" -mindepth 3 -maxdepth 3 -name plan.json -type f | wc -l)"
tasks="$(find "${root}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "tasks=${tasks} metadata=${metadata} plans=${plans}"
[[ "${tasks}" -eq 53 && "${metadata}" -eq 5300 && "${plans}" -eq 5300 ]]
