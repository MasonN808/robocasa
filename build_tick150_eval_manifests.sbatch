#!/usr/bin/env bash
#SBATCH --job-name=manifest-tick150
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
exec /work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python training/bc_task_vlm/build_tick100_eval_manifests.py --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent --preprocessed-dir /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick100_from150_train47_toolcall_idxnone --output-dir training/bc_task_vlm/eval_manifests/tick100_from150 --known-task-trajectories 10 --heldout-task-trajectories 10 --seed 1234
