#!/usr/bin/env bash
#SBATCH --job-name=tick-v9-failed-configs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_retry_v9_failed_configs}"
tasks=(ArrangeTea PlateStoreDinner PrepareCocktailStation PrepareSoupServing)
task="${tasks[SLURM_ARRAY_TASK_ID]}"
task_output="${output_root}/${task}"

cd "${repo}"
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true
mkdir -p "${task_output}"

echo "V9 exact failed-config retest: ${task}, run index 0"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${task}" \
  --num-runs 1 \
  --run-indices 0 \
  --random-start-location true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 1 \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --prompt-style simplified_v3 \
  --summary-path "${task_output}/summary.json"

echo "V9 exact failed-config retest finished at $(date --iso-8601=seconds)"
