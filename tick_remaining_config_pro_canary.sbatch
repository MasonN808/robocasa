#!/usr/bin/env bash
#SBATCH --job-name=tick-remain-pro
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
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_remaining_config_pro_canary
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
case "${SLURM_ARRAY_TASK_ID}" in
  0) task=CandleCleanup; run_index=0 ;;
  1) task=CandleCleanup; run_index=9 ;;
  2) task=OrganizeCondiments; run_index=4 ;;
  *) echo "Unexpected array index" >&2; exit 2 ;;
esac

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

run_dir="${output_root}/${task}_run_${run_index}"
mkdir -p "${run_dir}"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${task}" \
  --num-runs 12 \
  --run-indices "${run_index}" \
  --random-start-location true \
  --random-access-state true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3.1-pro-preview \
  --location global \
  --thinking-level high \
  --max-workers 1 \
  --max-retries 3 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --prompt-style simplified_v3 \
  --retry-feedback-style observational \
  --partition-policy none \
  --summary-path "${run_dir}/summary.json"
