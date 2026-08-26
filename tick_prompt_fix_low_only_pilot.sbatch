#!/usr/bin/env bash
#SBATCH --job-name=tick-low-pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-29%12
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
exemplar_root="${repo}/training/bc_task_vlm/reports/all_config_exemplar_bank"
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_prompt_fix_low_only_pilot}"
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
mapfile -t exemplars < <(find "${exemplar_root}" -name 'run_*.json' -type f | sort)
[[ "${#exemplars[@]}" -eq 824 ]]
flat_index=$((SLURM_ARRAY_TASK_ID * (${#exemplars[@]} - 1) / 29))
exemplar="${exemplars[${flat_index}]}"
task="$(basename "$(dirname "${exemplar}")")"
run_file="$(basename "${exemplar}")"
run_stem="${run_file#run_}"
run_stem="${run_stem%.json}"
run_index=$((10#${run_stem}))
num_runs="$(jq -r --arg task "${task}" '.tasks[] | select(.task == $task) | .possible_joint_configurations' "${manifest}")"

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

output="${output_root}/${task}/run_$(printf '%06d' "${run_index}").json"
mkdir -p "$(dirname "${output}")"
"${python_bin}" -m data_generation.task_level.generation.raw.cascade_canary \
  --task "${task}" \
  --run-index "${run_index}" \
  --num-runs "${num_runs}" \
  --exemplar "${exemplar}" \
  --output "${output}" \
  --low-direct-only
