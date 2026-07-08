#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/cleanup_after_push.sh <sweep_dir>

Removes 75% of traj_* directories under each task in <sweep_dir>,
keeping ~25% (ceil) for easy local access.

Optional environment overrides:
  KEEP_FRACTION=0.25
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

sweep_dir="$1"
keep_fraction="${KEEP_FRACTION:-0.25}"

if [[ ! -d "$sweep_dir" ]]; then
  echo "Sweep directory not found: $sweep_dir" >&2
  exit 1
fi

# Keep only task directories, skip .slurm_shards.
mapfile -t tasks < <(find "$sweep_dir" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

for task in "${tasks[@]}"; do
  if [[ "$task" == ".slurm_shards" ]]; then
    continue
  fi

  task_dir="$sweep_dir/$task"
  if [[ ! -d "$task_dir" ]]; then
    continue
  fi

  mapfile -t traj_dirs < <(find "$task_dir" -mindepth 1 -maxdepth 1 -type d -name "traj_*" -printf "%f\n" | sort)
  total_dirs=${#traj_dirs[@]}

  if [[ $total_dirs -eq 0 ]]; then
    echo "No traj_* directories under $task_dir"
    continue
  fi

  keep_count=$(python - <<EOF
import math
fraction = float("$keep_fraction")
print(max(1, math.ceil($total_dirs * fraction)))
EOF
)

  remove_count=$((total_dirs - keep_count))
  echo "Task $task: total $total_dirs, keeping $keep_count, removing $remove_count"

  for ((i=keep_count; i<total_dirs; i++)); do
    rm -rf "$task_dir/${traj_dirs[$i]}"
  done

done

echo "Cleanup complete for $sweep_dir"
