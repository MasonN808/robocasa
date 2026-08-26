#!/usr/bin/env bash
#SBATCH --job-name=tick-llm-critic
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
report_dir="${repo}/training/bc_task_vlm/reports/llm_critic_repair_canary"
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

"${python_bin}" -m data_generation.task_level.generation.raw.llm_critic_repair \
  --task "${task}" \
  --run-index "${run_index}" \
  --rejected-json "${report_dir}/inputs/${task}_run_${run_index}.json" \
  --output "${report_dir}/results/${task}_run_${run_index}.json" \
  --model gemini-3-flash-preview \
  --thinking-level high \
  --max-repairs 3
