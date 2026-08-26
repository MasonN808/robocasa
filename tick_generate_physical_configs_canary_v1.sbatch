#!/usr/bin/env bash
#SBATCH --job-name=tick-physical-canary-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --array=0-383%24
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/physical_configuration_canary_manifest_v1.json"
output_root="${repo}/training/bc_task_vlm/eval_runs/physical_configuration_canary_v1"
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
row="$(jq -c ".configurations[${SLURM_ARRAY_TASK_ID}]" "${manifest}")"
task="$(jq -r '.task' <<<"${row}")"
run_index="$(jq -r '.run_index' <<<"${row}")"
num_runs="$(jq -r '.num_runs' <<<"${row}")"
[[ -n "${task}" && "${task}" != null && "${run_index}" -ge 0 ]]

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
output="${output_root}/${task}/run_$(printf '%06d' "${run_index}").json"
mkdir -p "$(dirname "${output}")"

"${python_bin}" -m data_generation.task_level.generation.raw.cascade_canary \
  --task "${task}" \
  --run-index "${run_index}" \
  --num-runs "${num_runs}" \
  --output "${output}" \
  --location global \
  --temperature 0.6
