#!/usr/bin/env bash
#SBATCH --job-name=tick-post70sr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_structured_random_low
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_structured_random_low_image
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

cd "${repo}"
export PYTHONWARNINGS=ignore

mkdir -p "${output_root}"
ok=0
failed=0
for task_dir in "${input_root}"/*/; do
  task="$(basename "${task_dir}")"
  [[ -f "${task_dir}/summary.json" ]] || continue
  echo "postprocess ${task}"
  if "${python_bin}" -m data_generation.task_level.generation.image.cli \
      --dataset "${task_dir}/summary.json" \
      --output-dataset "${output_root}/${task}/summary.json" \
      --revalidate --disable-progress --workers 16; then
    ((ok+=1))
  else
    ((failed+=1))
    echo "FAILED ${task}" >&2
  fi
done

raw_count="$(find "${input_root}" -path '*/trajectories/traj_*.json' -type f | wc -l)"
processed_count="$(find "${output_root}" -path '*/trajectories/traj_*.json' -type f | wc -l)"
echo "postprocess tasks_ok=${ok} tasks_failed=${failed} raw=${raw_count} processed=${processed_count}"
[[ "${failed}" -eq 0 && "${processed_count}" -eq "${raw_count}" ]]
