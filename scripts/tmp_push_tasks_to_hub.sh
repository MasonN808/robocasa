#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
VENV="$REPO_ROOT/.venv"
IMAGE_DIR="$REPO_ROOT/data_generation/task_level/data/image/20260430T030150Z_full_run"
MERGED_SUMMARY="$IMAGE_DIR/add_lemon_to_fish/sweep_summary.json"
HF_USER="DorianAtSchool"
TIMESTAMP="20260430T030150Z_full_run"

echo "Activating venv: $VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
export MUJOCO_GL="egl"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before running this script}"

echo "Generating per-task sweep_summary.json files..."
python3 -c "
import json
from pathlib import Path
from collections import defaultdict

merged = json.load(open('$MERGED_SUMMARY'))
image_dir = Path('$IMAGE_DIR')

by_task = defaultdict(list)
for r in merged['results']:
    by_task[r['task_dir']].append(r)

for task_name, results in by_task.items():
    task_dir = image_dir / task_name
    if not task_dir.is_dir() or not list(task_dir.glob('traj_*')):
        print(f'Skipping {task_name} (no traj dirs)')
        continue
    summary = {**merged, 'results': results,
               'total': len(results),
               'succeeded': sum(1 for r in results if r['status'] == 'ok'),
               'failed': sum(1 for r in results if r['status'] == 'error')}
    out = task_dir / 'sweep_summary.json'
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'Written {out}')
"

MAX_PARALLEL=3
NUM_PROC=8  # threads per push (32 CPUs / 3 parallel tasks)
LOG_DIR="$REPO_ROOT/slurm_logs/push_logs_$$"
mkdir -p "$LOG_DIR"

echo ""
echo "Pushing each task to HuggingFace (MAX_PARALLEL=$MAX_PARALLEL, NUM_PROC=$NUM_PROC)..."

active_jobs=0
declare -a all_pids=()
declare -a all_tasks=()

for task_dir in "$IMAGE_DIR"/*/; do
    task=$(basename "$task_dir")
    [ -f "${task_dir}sweep_summary.json" ] || continue
    [ -d "${task_dir}traj_000000" ] || continue
    repo_id="${HF_USER}/robocasa_${TIMESTAMP}_${task}"
    log_file="$LOG_DIR/${task}.log"
    echo "=== Queuing $task -> $repo_id ==="
    (
        if python "$REPO_ROOT/scripts/push_sweep_to_hub.py" \
            --sweep-dir "$task_dir" \
            --repo-id "$repo_id" \
            --row-granularity trajectory \
            --num-proc "$NUM_PROC" \
            >"$log_file" 2>&1; then
            echo "OK" >"$log_file.status"
        else
            echo "FAILED" >"$log_file.status"
        fi
    ) &
    pid=$!
    all_pids+=("$pid")
    all_tasks+=("$task")
    ((++active_jobs))
    if ((active_jobs >= MAX_PARALLEL)); then
        wait -n || true
        ((--active_jobs))
    fi
done

wait

echo ""
echo "=== Results ==="
exit_code=0
for task in "${all_tasks[@]}"; do
    status_file="$LOG_DIR/${task}.log.status"
    status=$(cat "$status_file" 2>/dev/null || echo "UNKNOWN")
    if [[ "$status" == "OK" ]]; then
        echo "  OK:     $task"
    else
        echo "  FAILED: $task  (see $LOG_DIR/${task}.log)"
        exit_code=1
    fi
done

echo ""
echo "Per-task logs: $LOG_DIR"
echo "All done."
exit "$exit_code"
