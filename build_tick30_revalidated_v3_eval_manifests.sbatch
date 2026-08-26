#!/usr/bin/env bash
#SBATCH --job-name=manifest-tick30-v3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python training/bc_task_vlm/build_tick100_eval_manifests.py \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_revalidated_v3 \
  --preprocessed-dir /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_revalidated_v3_train47_toolcall_idxnone_schemafix \
  --output-dir training/bc_task_vlm/eval_manifests/tick30_revalidated_v3 \
  --known-task-trajectories 3 --heldout-task-trajectories 15 --seed 1234
