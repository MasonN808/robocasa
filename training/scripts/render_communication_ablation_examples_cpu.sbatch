#!/usr/bin/env bash
#SBATCH --job-name=gemini3-comm-videos-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONUNBUFFERED=1

exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python \
  training/bc_task_vlm/render_communication_ablation_examples.py
