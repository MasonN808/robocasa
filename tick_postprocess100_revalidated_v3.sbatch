#!/usr/bin/env bash
#SBATCH --job-name=tick-post100-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v3
output=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v3_image
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
mkdir -p "${output}"
for task_dir in "${input}"/*/; do
  task="$(basename "${task_dir}")"
  [[ -f "${task_dir}/summary.json" ]] || continue
  "${py}" -m data_generation.task_level.generation.image.cli --dataset "${task_dir}/summary.json" --output-dataset "${output}/${task}/summary.json" --revalidate --disable-progress --workers 16
done
raw="$(find "${input}" -path '*/trajectories/traj_*.json' -type f | wc -l)"
processed="$(find "${output}" -path '*/trajectories/traj_*.json' -type f | wc -l)"
valid="$(${py} -c 'import json,glob,sys; print(sum(bool(json.load(open(p)).get("validation",{}).get("is_valid")) for p in glob.glob(sys.argv[1]+"/*/trajectories/traj_*.json")))' "${output}")"
echo "raw=${raw} processed=${processed} valid=${valid}"
[[ "${raw}" -eq 5300 && "${processed}" -eq 5300 && "${valid}" -eq 5300 ]]
