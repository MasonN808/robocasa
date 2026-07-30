#!/usr/bin/env bash
#SBATCH --job-name=push-v1_5-verbalized-sampling-hf
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Pushes the clean v1.5 reasoning-SFT LoRA adapter, trained on the
# verbalized_sampling data source (robocasa_local_train_subset_21, v1's
# actual training data), to the HF Hub.
#   sbatch --dependency=afterok:<train_job_id> push_v1_5_verbalized_sampling_hf.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache

"${PYTHON_BIN}" push_adapter_hf.py \
  --adapter-dir training/bc_task_vlm/runs/qwen3vl-8b-v1_5-verbalized-sampling \
  --repo-id DorianAtSchool/qwen3vl-8b-robocasa-v1_5-verbalized-sampling-sft

echo "Done."
