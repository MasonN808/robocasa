#!/usr/bin/env bash
#SBATCH --job-name=tick-critic-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-27%12
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
exemplar_root="${repo}/training/bc_task_vlm/reports/all_config_exemplar_bank"
source_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_all_config_cascade_canary_v2
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_critic_misses_retry_v3
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
mapfile -t cases < <(
  find "${source_root}" -mindepth 2 -maxdepth 2 -name 'run_*.json' -print0 \
    | xargs -0 jq -r 'select(([.attempts[]? | select(.stage == "flash_critic_repair")] | length) > 0 and .accepted_stage != "flash_critic_repair") | "\(.task):\(.run_index)"' \
    | sort -t: -k1,1 -k2,2n
)
[[ "${#cases[@]}" -eq 28 ]]
case_value="${cases[${SLURM_ARRAY_TASK_ID}]}"
task="${case_value%%:*}"
run_index="${case_value##*:}"
num_runs="$(jq -r --arg task "${task}" '.tasks[] | select(.task == $task) | .possible_joint_configurations' "${manifest}")"
[[ -n "${task}" && "${num_runs}" -gt 0 ]]

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

task_dir="${output_root}/${task}"
mkdir -p "${task_dir}"
output="${task_dir}/run_$(printf '%06d' "${run_index}").json"
"${python_bin}" -m data_generation.task_level.generation.raw.cascade_canary \
  --task "${task}" \
  --run-index "${run_index}" \
  --num-runs "${num_runs}" \
  --exemplar "${exemplar_root}/${task}/run_$(printf '%06d' "${run_index}").json" \
  --output "${output}"
