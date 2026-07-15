#!/usr/bin/env bash
#SBATCH --job-name=robocasa-tar-extracted
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# Tars an already fully staged-extracted task (from the interrupted staging
# array) into staged_tars/<task>.tar, verifies the member count, then deletes
# the extracted copy to reclaim inodes on /work.
set -euo pipefail
tasks=(alcohol_serving_prep align_silverware arrange_bread_bowl arrange_drinkware
       bowl_and_cup cheese_mixing colorful_salsa condiment_collection
       cut_buffet_pizza date_night deliver_straw dessert_assembly
       display_meat_variety distribute_chicken)
task="${tasks[${SLURM_ARRAY_TASK_ID:?}]}"
src="training/bc_task_vlm/staged_hf/experiment/${task}"
out="training/bc_task_vlm/staged_tars/${task}.tar"
mkdir -p training/bc_task_vlm/staged_tars
[[ -f "${out}" ]] && { echo "${out} exists; skipping"; exit 0; }
expected=$(find "${src}/${task}" -name adapted_trajectory.json | wc -l)
(( expected == 3000 )) || { echo "ERROR: ${task} has ${expected} complete trajectories"; exit 1; }
tar -cf "${out}.partial" -C "${src}" "${task}"
members=$(tar -tf "${out}.partial" | grep -c adapted_trajectory.json)
(( members == expected )) || { echo "ERROR: tar members ${members} != ${expected}"; exit 1; }
mv "${out}.partial" "${out}"
rm -rf "${src}"
echo "Wrote ${out} ($(du -h "${out}" | cut -f1)); removed extracted copy"
