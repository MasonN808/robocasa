#!/usr/bin/env bash
#SBATCH --job-name=push-v3-hf
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export HF_HOME=/work/umass/shlomo_umass/hf_cache
export HF_TOKEN=$(cat /work/umass/shlomo_umass/hf_cache/token)

/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python push_v3_hf.py
