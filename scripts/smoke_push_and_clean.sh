#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
VENV="$REPO_ROOT/.venv"

usage() {
  cat <<'EOF'
Usage: bash scripts/smoke_push_and_clean.sh <sweep_dir> <hf_repo_id> [keep_fraction]

Runs a smoke push of a single sweep directory to HF and then runs cleanup.

Arguments:
  <sweep_dir>    Path to the sweep directory (task-level image dir)
  <hf_repo_id>   Hugging Face repo id (user/repo)
  [keep_fraction] Optional fraction to keep locally (default 0.25)

Environment:
  The script will activate the project's virtualenv at $REPO_ROOT/.venv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

sweep_dir="$1"
repo_id="$2"
keep_fraction="${3:-0.25}"

if [[ ! -d "$sweep_dir" ]]; then
  echo "Sweep directory not found: $sweep_dir" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "Virtualenv not found at $VENV" >&2
  exit 1
fi

echo "Activating venv: $VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "Starting push for $sweep_dir -> $repo_id"
start_ts=$(date +%s)
python "$REPO_ROOT/scripts/push_sweep_to_hub.py" --sweep-dir "$sweep_dir" --repo-id "$repo_id" --row-granularity trajectory
push_exit=$?
end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))

if [[ $push_exit -ne 0 ]]; then
  echo "Push failed with exit code $push_exit"
  exit $push_exit
fi

echo "Push succeeded in ${elapsed}s. Running cleanup (keep_fraction=${keep_fraction})"

# Export KEEP_FRACTION for the cleanup helper
KEEP_FRACTION="$keep_fraction" bash "$REPO_ROOT/scripts/cleanup_after_push.sh" "$sweep_dir"

echo "Smoke push+clean completed for $sweep_dir"
