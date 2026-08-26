#!/usr/bin/env bash
#SBATCH --job-name=tick-lite-ab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-105%6
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_flash_lite_ab
models=(gemini-3.5-flash-lite gemini-3.1-flash-lite)
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
task_index=$((SLURM_ARRAY_TASK_ID / 2))
model_index=$((SLURM_ARRAY_TASK_ID % 2))
model="${models[${model_index}]}"
task="$(jq -r ".tasks[${task_index}].task" "${manifest}")"
num_runs="$(jq -r ".tasks[${task_index}].possible_joint_configurations" "${manifest}")"
[[ -n "${task}" && "${task}" != null && "${num_runs}" -gt 0 ]]

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

task_dir="${output_root}/${model}/${task}"
mkdir -p "${task_dir}"
for run_index in $(seq 0 $((num_runs - 1))); do
  output="${task_dir}/run_$(printf '%06d' "${run_index}").json"
  if [[ -s "${output}" ]]; then
    continue
  fi
  "${python_bin}" -m data_generation.task_level.generation.raw.thinking_level_canary \
    --task "${task}" \
    --run-index "${run_index}" \
    --num-runs "${num_runs}" \
    --model "${model}" \
    --thinking-level low \
    --output "${output}"
done
