#!/usr/bin/env bash
#SBATCH --job-name=robocasa-stage-tar
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

# Stage one full HF sweep dataset per array task as a SINGLE tar on /work.
#
# The /work group quota is inode-limited (~3.5M files free; one extracted task
# is ~120k files), so nothing extracted ever lands on /work here: the dataset
# is downloaded + extracted on node-local /tmp, tarred to
# training/bc_task_vlm/staged_tars/<task>.tar (1 file, ~9 GB), and /tmp is
# wiped on exit. Training/eval jobs untar to their own node-local storage via
# training/scripts/assemble_dataset_root.sh.
#
# Submit with:
#   sbatch --array=0-<N-1> training/scripts/stage_experiment_tasks_to_tar.sh

set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
repo_list_file="${HF_DATASET_REPOS_FILE:-${repo_root}/training/bc_task_vlm/hf_repo_lists/experiment_missing_train_tasks.txt}"
tar_root="${HF_STAGE_TAR_ROOT:-${repo_root}/training/bc_task_vlm/staged_tars}"
stage_workers="${HF_STAGE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
max_episodes="${HF_STAGE_MAX_EPISODES:-}"  # empty = full dataset

work_dir="/tmp/stage_tar_${SLURM_JOB_ID:-$$}"
export HF_HOME="${work_dir}/hf_home"
mkdir -p "${work_dir}" "${HF_HOME}" "${tar_root}"
trap 'rm -rf "${work_dir}"' EXIT

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
tar_path="${tar_root}/${task_name}.tar"

if [[ -f "${tar_path}" ]]; then
  echo "${tar_path} already exists ($(du -h "${tar_path}" | cut -f1)); skipping."
  exit 0
fi

extra_args=()
if [[ -n "${max_episodes}" ]]; then
  extra_args+=(--max-total-episodes "${max_episodes}")
fi

echo "Staging ${repo_id} -> ${work_dir}/stage (workers=${stage_workers} max_episodes=${max_episodes:-all})"
"${python_bin}" "${repo_root}/training/scripts/stage_hf_sweep_datasets.py" \
  --repo-id "${repo_id}" \
  --output-root "${work_dir}/stage" \
  --workers "${stage_workers}" \
  --resume \
  "${extra_args[@]}"

if [[ ! -d "${work_dir}/stage/${task_name}" ]]; then
  echo "ERROR: expected ${work_dir}/stage/${task_name} after staging" >&2
  ls "${work_dir}/stage" >&2 || true
  exit 1
fi

traj_count=$(ls "${work_dir}/stage/${task_name}" | wc -l)
echo "Staged ${traj_count} trajectories; creating tar..."

# Write to a temp name so a partially written tar is never mistaken for done.
tar -cf "${tar_path}.partial" -C "${work_dir}/stage" "${task_name}"
tar_members=$(tar -tf "${tar_path}.partial" | grep -c "adapted_trajectory.json" || true)
if (( tar_members < traj_count )); then
  echo "ERROR: tar has ${tar_members} trajectory.json members, expected ${traj_count}" >&2
  exit 1
fi
mv "${tar_path}.partial" "${tar_path}"
echo "Wrote ${tar_path} ($(du -h "${tar_path}" | cut -f1), ${tar_members} trajectories)"
