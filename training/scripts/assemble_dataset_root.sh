#!/usr/bin/env bash
# Assemble the 52-task experiment dataset root for a job, without putting
# extracted files on inode-limited /work:
#   - tasks already extracted on /work (the original local sweep + small
#     eval-only staged tasks) are symlinked in place;
#   - tar-staged tasks (training/bc_task_vlm/staged_tars/*.tar) are untarred
#     to the target root, which should be node-local storage (/tmp) inside a
#     SLURM job.
#
# Usage:
#   assemble_dataset_root.sh <target_root> [--tars-only task1,task2,...]
#
# Prints the target root on success so callers can do:
#   DATA_ROOT=$(bash training/scripts/assemble_dataset_root.sh /tmp/exp_root)

set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
local_sweep="${repo_root}/data_generation/task_level/data/image/20260430T030150Z_full_run"
staged_extracted="${repo_root}/training/bc_task_vlm/staged_hf/experiment"
tar_root="${repo_root}/training/bc_task_vlm/staged_tars"
task_list="${repo_root}/data_analysis/analysis/held_out_task_selection/experiment_tasks.txt"

target_root="${1:?usage: assemble_dataset_root.sh <target_root>}"
only_tasks=""
if [[ "${2:-}" == "--tars-only" ]]; then
  only_tasks="${3:?--tars-only needs a comma-separated task list}"
fi

mkdir -p "${target_root}"

linked=0 untarred=0 missing=()
while read -r task; do
  [[ -z "${task}" ]] && continue
  if [[ -n "${only_tasks}" && ",${only_tasks}," != *",${task},"* ]]; then
    continue
  fi
  dest="${target_root}/${task}"
  if [[ -e "${dest}" ]]; then
    linked=$((linked + 1))
    continue
  fi
  # The root layout is <root>/<task>/traj_*: local sweep tasks are already
  # flat, staged-extracted tasks nest one level, tars contain <task>/traj_*.
  if [[ -d "${local_sweep}/${task}" ]]; then
    ln -s "${local_sweep}/${task}" "${dest}"
    linked=$((linked + 1))
  elif [[ -f "${tar_root}/${task}.tar" ]]; then
    tar -xf "${tar_root}/${task}.tar" -C "${target_root}"
    untarred=$((untarred + 1))
  elif [[ -d "${staged_extracted}/${task}/${task}" ]]; then
    ln -s "${staged_extracted}/${task}/${task}" "${dest}"
    linked=$((linked + 1))
  else
    missing+=("${task}")
  fi
done < "${task_list}"

echo "Assembled ${target_root}: ${linked} symlinked, ${untarred} untarred" >&2
if (( ${#missing[@]} > 0 )); then
  echo "MISSING (${#missing[@]}): ${missing[*]}" >&2
  exit 2
fi
echo "${target_root}"
