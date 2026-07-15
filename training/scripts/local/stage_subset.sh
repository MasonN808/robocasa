#!/usr/bin/env bash
# LOCAL. Downloads N trajectories/task for the 47 training tasks directly from
# HuggingFace into a subset dataset root — no cluster and no rsync needed.
#
# Output layout (what training/bc_task_vlm/main.py expects):
#   <SUBSET_ROOT>/<task>/traj_XXXXXX/{adapted_trajectory.json,images,...}
#
# --max-total-episodes takes the first N episodes per repo (deterministic), so
# the subset matches what the cluster tars contain. ~4 GB at 30 traj/task.
#
# Prereqs: a Python env with this repo importable plus huggingface_hub /
# datasets / Pillow. Set HF_TOKEN if the datasets are gated.
#
# Usage:
#   TRAJ_PER_TASK=30 SUBSET_ROOT=~/robocasa_local_train_subset \
#     bash training/scripts/local/stage_subset.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

traj_per_task="${TRAJ_PER_TASK:-30}"
subset_root="${SUBSET_ROOT:-${HOME}/robocasa_local_train_subset}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"
hf_prefix="${HF_REPO_PREFIX:-DorianAtSchool/robocasa_20260430T030150Z_full_run_}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

read_tasks='import json,sys; print("\n".join(json.load(open(sys.argv[1]))["train_tasks"]))'
mapfile -t train_tasks < <("${python_bin}" -c "${read_tasks}" "${selection}")

mkdir -p "${subset_root}"
echo "Downloading ${traj_per_task} traj/task for ${#train_tasks[@]} tasks -> ${subset_root}"

for task in "${train_tasks[@]}"; do
  echo "=== ${task} ==="
  "${python_bin}" training/scripts/stage_hf_sweep_datasets.py \
    --repo-id "${hf_prefix}${task}" \
    --output-root "${subset_root}" \
    --max-total-episodes "${traj_per_task}" \
    --workers "${WORKERS:-8}" \
    --resume
done

echo ""
echo "Subset ready at ${subset_root} ($(du -sh "${subset_root}" 2>/dev/null | cut -f1))"
echo "Next: probe throughput, then train:"
echo "  DATA_ROOT=${subset_root} bash training/scripts/local/probe_throughput.sh"
echo "  DATA_ROOT=${subset_root} bash training/scripts/local/train_qwen3vl_8b.sh"
