#!/usr/bin/env bash
#SBATCH --job-name=tick-render-incr-v3
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
input=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x100_from150_partition_none_temp06_v3_image_affected
output=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mapfile -t all < <(find "${input}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
tasks=(); for i in "${!all[@]}"; do (( i % 8 == SLURM_ARRAY_TASK_ID )) && tasks+=("${all[i]}"); done
if (( ${#tasks[@]} )); then
  "${py}" scripts/sweep_trajectories.py --input-dir "${input}" --output-dir "${output}" --tasks "${tasks[@]}" --step-order concurrent --gpu-ids 0 --gl-backend egl
fi
expected=0
for task in "${tasks[@]}"; do
  count="$(${py} -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["trajectory_files"]))' "${input}/${task}/summary.json")"
  expected=$(( expected + count ))
done
metadata=0; plans=0
for task in "${tasks[@]}"; do
  metadata=$(( metadata + $(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name trajectory_execution_metadata.json -type f | wc -l) ))
  plans=$(( plans + $(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name plan.json -type f | wc -l) ))
done
echo "incremental render gate shard=${SLURM_ARRAY_TASK_ID}: expected_selected_task_totals=${expected} metadata_all_selected=${metadata} plans_all_selected=${plans}"
# Each task directory contains reused plus affected outputs, so its final count
# must equal the selected count (100) once this shard completes its affected IDs.
for task in "${tasks[@]}"; do
  [[ "$(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name trajectory_execution_metadata.json -type f | wc -l)" -eq 100 ]]
  [[ "$(find "${output}/${task}" -mindepth 2 -maxdepth 2 -name plan.json -type f | wc -l)" -eq 100 ]]
done
