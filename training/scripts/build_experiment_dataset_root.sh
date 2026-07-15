#!/usr/bin/env bash
# Builds the combined 52-task dataset root for the SFT-necessity experiment as
# a directory of per-task symlinks: local rendered tasks come from the
# 20260430T030150Z_full_run root; tasks staged from HF come from
# training/bc_task_vlm/staged_hf/experiment/<task>/<task>.
# Idempotent — re-run after more tasks finish staging.
set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
local_root="${repo_root}/data_generation/task_level/data/image/20260430T030150Z_full_run"
staged_root="${repo_root}/training/bc_task_vlm/staged_hf/experiment"
combined_root="${EXPERIMENT_DATASET_ROOT:-${repo_root}/data_generation/task_level/data/image/experiment_52}"

mkdir -p "${combined_root}"

selection="${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json"
mapfile -t tasks < <("${repo_root}/.venv/bin/python" - <<EOF
import json
selection = json.load(open("${selection}"))
for task in selection["held_out_tasks"] + selection["train_tasks"]:
    print(task)
EOF
)

linked=0
missing=()
for task in "${tasks[@]}"; do
  target=""
  if [[ -d "${local_root}/${task}" ]]; then
    target="${local_root}/${task}"
  elif [[ -d "${staged_root}/${task}/${task}" ]]; then
    target="${staged_root}/${task}/${task}"
  fi
  if [[ -z "${target}" ]]; then
    missing+=("${task}")
    continue
  fi
  ln -sfn "${target}" "${combined_root}/${task}"
  ((++linked))
done

echo "Linked ${linked}/${#tasks[@]} tasks into ${combined_root}"
if ((${#missing[@]} > 0)); then
  echo "Still missing (staging incomplete): ${missing[*]}"
  exit 2
fi
