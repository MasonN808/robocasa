#!/usr/bin/env bash
#SBATCH --job-name=tick-prod-cascade-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
tasks=(ArrangeBreadBowl PrepareCoffee SetUpSpiceStation HotDogSetup GarnishCake MeatSkewerAssembly)
task="${tasks[SLURM_ARRAY_TASK_ID]}"
cd "${repo}"
[[ "${CONFIRM_SPEND:-}" == yes ]]
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONPATH="${repo}"
exec "${python_bin}" -m data_generation.task_level.generation.raw.production_cascade \
  --task "${task}" --num-runs 1 \
  --output-root training/bc_task_vlm/eval_runs/production_cascade_smoke_raw
