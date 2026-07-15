#!/usr/bin/env bash
#SBATCH --job-name=robocasa-stage-experiment
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# Stage one HF sweep dataset per array task into per-task roots under
# training/bc_task_vlm/staged_hf/experiment/<task>/ for the SFT-necessity
# experiment. Submit with:
#   sbatch --array=0-<N-1> training/scripts/stage_experiment_tasks_array.sh
# where N = number of lines in the repo list.

set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
repo_list_file="${HF_DATASET_REPOS_FILE:-${repo_root}/training/bc_task_vlm/hf_repo_lists/experiment_missing_train_tasks.txt}"
stage_root="${HF_STAGE_ROOT:-${repo_root}/training/bc_task_vlm/staged_hf/experiment}"
stage_workers="${HF_STAGE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
# Disk quota on /work is limited: default to 400 episodes/task (~1.2 GB)
# instead of the full 3000 (~9.3 GB). Override explicitly if more is needed.
max_episodes="${HF_STAGE_MAX_EPISODES:-400}"

# Keep the parquet download cache off /work entirely: use a job-owned
# node-local /tmp dir. Never accept an inherited HF_HOME here — the cleanup
# trap must only ever delete a directory this job created (an inherited value
# could be the shared group cache).
stage_cache_dir="/tmp/hf_stage_cache_${SLURM_JOB_ID:-$$}"
export HF_HOME="${stage_cache_dir}"
mkdir -p "${stage_cache_dir}"
trap 'rm -rf "${stage_cache_dir}"' EXIT

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${repo_root}/.env"
  set +a
fi

mapfile -t repos < <(grep -v '^\s*$' "${repo_list_file}")
index="${SLURM_ARRAY_TASK_ID:?Submit as an sbatch array over the repo list}"
if (( index >= ${#repos[@]} )); then
  echo "Array index ${index} beyond repo list size ${#repos[@]}; nothing to do."
  exit 0
fi

repo_id="${repos[index]}"
task_name="${repo_id##*_full_run_}"
output_root="${stage_root}/${task_name}"
mkdir -p "${output_root}"

extra_args=()
if [[ -n "${max_episodes}" ]]; then
  extra_args+=(--max-total-episodes "${max_episodes}")
fi

echo "Staging ${repo_id} -> ${output_root} (workers=${stage_workers} max_episodes=${max_episodes:-all})"
exec "${python_bin}" "${repo_root}/training/scripts/stage_hf_sweep_datasets.py" \
  --repo-id "${repo_id}" \
  --output-root "${output_root}" \
  --workers "${stage_workers}" \
  --resume \
  "${extra_args[@]}"
