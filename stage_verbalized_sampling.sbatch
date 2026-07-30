#!/usr/bin/env bash
#SBATCH --job-name=stage-verbalized-sampling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Downloads robocasa_local_train_subset_21 (v1's real training data) and
# robocasa-exp52-eval (v1's real eval manifests + data) from HF. Moved off
# the login node: that session's memory cgroup is capped at 5GB (shared with
# the interactive CLI session), and dmesg confirmed the OOM killer was
# repeatedly reaping the login-node hf download processes once that filled
# up -- not an HF-side issue. Compute nodes get a real --mem allocation
# instead.
#   sbatch stage_verbalized_sampling.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export HF_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/hf
export HF_HOME=/work/umass/shlomo_umass/hf_cache
# Xet backend was flaky earlier (429s, then a CDN 500) -- plain HTTP is
# slower but more reliable.
export HF_HUB_DISABLE_XET=1

echo "=== robocasa_local_train_subset_21 ==="
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
"${HF_BIN}" download DorianAtSchool/robocasa_local_train_subset_21 --repo-type dataset \
  --local-dir /work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21 \
  --max-workers 2

echo "=== robocasa-exp52-eval ==="
"${HF_BIN}" download DorianAtSchool/robocasa-exp52-eval --repo-type dataset \
  --local-dir /work/umass/shlomo_umass/dbenhamougol_umass/robocasa \
  --max-workers 2

echo "Done."
