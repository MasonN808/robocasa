#!/usr/bin/env bash
#SBATCH --job-name=push-local-only-to-hub
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougol@umass.edu

set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${repo_root}/.env"
  set +a
fi

bash "${repo_root}/scripts/tmp_push_local_only_tasks_to_hub.sh"
