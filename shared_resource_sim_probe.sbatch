#!/usr/bin/env bash
#SBATCH --job-name=shared-resource-probe
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
manifest=training/bc_task_vlm/eval_manifests/diagnostics/shared_resource_sim_probe.json
dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
output_dir=training/bc_task_vlm/eval_runs/shared_resource_sim_probe

exec "${python_bin}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle \
  --manifest "${manifest}" \
  --dataset-root "${dataset_root}" \
  --output-dir "${output_dir}" \
  --seed 12011 \
  --gl-backend egl \
  --map-renderer raster \
  --map-dpi 60 \
  --train-get-image \
  --partial-history \
  --partial-step-index-mode none \
  --partial-observation-mode consume-once \
  --uniform-durations \
  --concurrent-expert-replay \
  --success-criterion fsm \
  --no-two-pass-views
