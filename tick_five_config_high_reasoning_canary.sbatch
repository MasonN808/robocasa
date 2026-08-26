#!/usr/bin/env bash
#SBATCH --job-name=tick-5cfg-high
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_five_config_high_reasoning_canary
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
case "${SLURM_ARRAY_TASK_ID}" in
  0) task=CandleCleanup; run_indices=(0 1 9) ;;
  1) task=CondimentCollection; run_indices=(6) ;;
  2) task=OrganizeCondiments; run_indices=(4) ;;
  *) echo "Unexpected array index" >&2; exit 2 ;;
esac

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

task_dir="${output_root}/${task}"
mkdir -p "${task_dir}"
common=(
  --tasks "${task}"
  --num-runs 12
  --random-start-location true
  --random-access-state true
  --sampling structured_random
  --temperature 0.6
  --model gemini-3-flash-preview
  --location global
  --thinking-level high
  --max-workers 3
  --max-retries 1
  --generation-timeout-sec 600
  --enable-validation
  --tick-format
  --prompt-style simplified_v3
  --retry-feedback-style observational
  --partition-policy none
)

# Three independent invocations. Each resumed invocation advances the attempt
# seed but begins from the canonical prompt, without a rejected trajectory or
# accumulated repair text from the preceding invocation.
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  "${common[@]}" --run-indices "${run_indices[@]}" \
  --summary-path "${task_dir}/summary.json" || true
for _round in 2 3; do
  [[ -s "${task_dir}/summary.json" ]] || exit 1
  if jq -e '.pending_run_indices | length == 0' "${task_dir}/summary.json" >/dev/null; then
    break
  fi
  "${python_bin}" -m data_generation.task_level.generation.raw.cli \
    "${common[@]}" --resume "${task_dir}" || true
done

jq -e '.pending_run_indices | length == 0' "${task_dir}/summary.json"
