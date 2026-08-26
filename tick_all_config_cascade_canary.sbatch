#!/usr/bin/env bash
#SBATCH --job-name=tick-cascade-824
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-52%12
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
exemplar_root="${repo}/training/bc_task_vlm/reports/all_config_exemplar_bank"
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_all_config_cascade_canary
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
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
if [[ -n "${PILOT_RUN_INDEX:-}" ]]; then
  indices=("${PILOT_RUN_INDEX}")
else
  mapfile -t indices < <(seq 0 $((num_runs - 1)))
fi

failed=0
for run_index in "${indices[@]}"; do
  output="${task_dir}/run_$(printf '%06d' "${run_index}").json"
  if [[ -s "${output}" ]] && jq -e '.is_valid == true' "${output}" >/dev/null; then
    continue
  fi
  "${python_bin}" -m data_generation.task_level.generation.raw.cascade_canary \
    --task "${task}" \
    --run-index "${run_index}" \
    --num-runs "${num_runs}" \
    --exemplar "${exemplar_root}/${task}/run_$(printf '%06d' "${run_index}").json" \
    --output "${output}" || failed=1
done
exit "${failed}"
