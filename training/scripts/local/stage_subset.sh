#!/usr/bin/env bash
# CLUSTER-SIDE. Builds a small N-trajectories-per-task training subset of the
# 47-task experiment dataset so it can be rsync'd to a local GPU box (e.g. an
# RTX 5090) and fine-tuned without the full ~170 GB corpus.
#
# Output layout (what training/bc_task_vlm/main.py expects):
#   <SUBSET_ROOT>/<task>/traj_XXXXXX/{adapted_trajectory.json,images,...}
#
# Sources, per task:
#   - tar-staged tasks: extracted from training/bc_task_vlm/staged_tars/<task>.tar
#   - local-sweep tasks: copied from the extracted 20260430 sweep
# Trajectories are taken in sorted order (deterministic) so re-running is stable.
#
# Usage (on the cluster):
#   TRAJ_PER_TASK=30 bash training/scripts/local/stage_subset.sh
set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

traj_per_task="${TRAJ_PER_TASK:-30}"
subset_root="${SUBSET_ROOT:-${repo_root}/training/bc_task_vlm/local_train_subset}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"
local_sweep="${repo_root}/data_generation/task_level/data/image/20260430T030150Z_full_run"
tar_root="${repo_root}/training/bc_task_vlm/staged_tars"

mapfile -t train_tasks < <("${python_bin}" - "${selection}" <<'EOF'
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["train_tasks"]))
EOF
)

mkdir -p "${subset_root}"
echo "Staging ${traj_per_task} traj/task for ${#train_tasks[@]} tasks -> ${subset_root}"

staged=0 missing=()
for task in "${train_tasks[@]}"; do
  dest="${subset_root}/${task}"
  mkdir -p "${dest}"
  if [[ -f "${tar_root}/${task}.tar" ]]; then
    # First N trajectory directories inside the tar (members are <task>/traj_*/).
    mapfile -t members < <(
      tar -tf "${tar_root}/${task}.tar" \
        | grep -oE "^${task}/traj_[0-9]+/" | sort -u | head -n "${traj_per_task}"
    )
    # tar members already carry the <task>/ prefix, so extracting at the subset
    # root yields <subset_root>/<task>/traj_*, exactly the expected layout.
    tar -xf "${tar_root}/${task}.tar" -C "${subset_root}" "${members[@]}"
  elif [[ -d "${local_sweep}/${task}" ]]; then
    mapfile -t traj_dirs < <(
      find "${local_sweep}/${task}" -maxdepth 1 -type d -name 'traj_*' \
        | sort | head -n "${traj_per_task}"
    )
    for traj in "${traj_dirs[@]}"; do
      cp -r "${traj}" "${dest}/"
    done
  else
    missing+=("${task}")
    continue
  fi
  count=$(find "${dest}" -maxdepth 1 -type d -name 'traj_*' | wc -l)
  staged=$((staged + 1))
  echo "  ${task}: ${count} trajectories"
done

echo ""
echo "Staged ${staged}/${#train_tasks[@]} tasks into ${subset_root}"
if (( ${#missing[@]} > 0 )); then
  echo "MISSING (no tar or local sweep): ${missing[*]}" >&2
fi
subset_size=$(du -sh "${subset_root}" 2>/dev/null | cut -f1 || echo "?")
echo "Subset size: ${subset_size}"
echo ""
echo "Next: pull it to your local GPU box, e.g."
echo "  rsync -avhP --info=progress2 \\"
echo "    <cluster-host>:${subset_root}/ \\"
echo "    ~/robocasa_local_train_subset/"
echo "then run training/scripts/local/train_qwen3vl_8b.sh there."
