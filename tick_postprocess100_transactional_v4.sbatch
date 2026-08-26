#!/usr/bin/env bash
#SBATCH --job-name=tick-post100-v4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional
output=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v4_transactional_image
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
mkdir -p "${output}"
for task_dir in "${input}"/*/; do
  task="$(basename "${task_dir}")"
  [[ -f "${task_dir}/summary.json" ]] || continue
  "${py}" -m data_generation.task_level.generation.image.cli \
    --dataset "${task_dir}/summary.json" \
    --output-dataset "${output}/${task}/summary.json" \
    --revalidate --disable-progress --workers 16
done
processed="$(find "${output}" -path '*/trajectories/traj_*.json' -type f | wc -l)"
valid="$(${py} -c 'import json,glob,sys; print(sum(json.load(open(p)).get("validation",{}).get("is_valid") is True and json.load(open(p)).get("validation",{}).get("validator_contract_version")=="transactional_tick_v1" for p in glob.glob(sys.argv[1]+"/*/trajectories/traj_*.json")))' "${output}")"
echo "processed=${processed} current_valid=${valid}"
[[ "${processed}" -eq 5300 && "${valid}" -eq 5300 ]]
