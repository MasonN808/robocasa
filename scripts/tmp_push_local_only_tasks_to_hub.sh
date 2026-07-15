#!/usr/bin/env bash
# Push the 3 verified tasks that exist only in the local rendered dataset
# (never published) so all 52 verified tasks live on the Hub.
set -euo pipefail

REPO_ROOT="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
VENV="$REPO_ROOT/.venv"
IMAGE_DIR="$REPO_ROOT/data_generation/task_level/data/image/20260430T030150Z_full_run"
HF_USER="DorianAtSchool"
TIMESTAMP="20260430T030150Z_full_run"

TASKS=(
    line_up_condiments
    prepare_cocktail_station
    set_bowls_for_soup
)

echo "Activating venv: $VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
export MUJOCO_GL="egl"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before running this script}"

MAX_PARALLEL=3
NUM_PROC=8
LOG_DIR="$REPO_ROOT/slurm_logs/push_logs_local_only_$$"
mkdir -p "$LOG_DIR"

echo ""
echo "Pushing local-only tasks to HuggingFace (MAX_PARALLEL=$MAX_PARALLEL, NUM_PROC=$NUM_PROC)..."

active_jobs=0
declare -a all_pids=()
declare -a all_tasks=()

for task in "${TASKS[@]}"; do
    task_dir="$IMAGE_DIR/$task/"
    [ -f "${task_dir}sweep_summary.json" ] || { echo "Skipping $task (no sweep_summary.json)"; continue; }
    [ -d "${task_dir}traj_000000" ] || { echo "Skipping $task (no traj_000000)"; continue; }
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
