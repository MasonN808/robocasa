#!/usr/bin/env bash
#SBATCH --job-name=download-kitchen-assets
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# robocasa/models/assets/objects/ is entirely gitignored (large binaries,
# provisioned separately per docs/introduction/installation.md) and was
# missing from this checkout entirely -- live-sim eval's oracle gate failed
# with FileNotFoundError building the default scene (layout=11/style=34)
# because of it. This is the documented fix (~10GB).
#   sbatch download_kitchen_assets.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

echo y | "${PYTHON_BIN}" -m robocasa.scripts.download_kitchen_assets --type all

echo "Done."
