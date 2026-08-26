#!/usr/bin/env bash
#SBATCH --job-name=tick-atomic-rule
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_atomic_rule_canary}"
tasks=(CandleCleanup OrganizeCondiments PlateStoreDinner PrepareCocktailStation PrepareCoffee)
run_indices=(0 0 0 1 0)
slot="${SLURM_ARRAY_TASK_ID}"
task="${tasks[slot]}"
run_index="${run_indices[slot]}"
task_slug="$(printf '%s' "${task}" | sed -E 's/([a-z0-9])([A-Z])|([A-Z])([A-Z][a-z])/\1\3_\2\4/g' | tr '[:upper:]' '[:lower:]')"
task_output="${output_root}/${task_slug}/run_${run_index}"

cd "${repo}"
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true
mkdir -p "${task_output}"

"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${task}" \
  --num-runs "$((run_index + 1))" \
  --run-indices "${run_index}" \
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
