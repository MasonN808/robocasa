#!/usr/bin/env bash
#SBATCH --job-name=tick-allcfg-canary-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-52%12
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick53_all_configs_canary_state_grounded_v1}"
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
[[ -s "${manifest}" ]]
task="$(jq -r ".tasks[${SLURM_ARRAY_TASK_ID}].task" "${manifest}")"
num_runs="$(jq -r ".tasks[${SLURM_ARRAY_TASK_ID}].possible_joint_configurations" "${manifest}")"
[[ -n "${task}" && "${task}" != null && "${num_runs}" -gt 0 ]]

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

task_dir="${output_root}/${task}"
mkdir -p "${task_dir}"
echo "all-config canary: task=${task} configurations=${num_runs}"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${task}" \
  --num-runs "${num_runs}" \
  --random-start-location true \
  --random-access-state true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --prompt-style simplified_v3 \
  --retry-feedback-style targeted \
  --partition-policy none \
  --summary-path "${task_dir}/summary.json"
