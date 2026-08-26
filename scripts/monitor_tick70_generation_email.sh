#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 SLURM_ARRAY_JOB_ID" >&2
  exit 2
fi

JOB_ID="$1"
EMAIL_TO="${EMAIL_TO:-dbenhamougol@umass.edu}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-RoboCasa tick70 structured-random generation}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_structured_random_low}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/slurm_logs/tick70_structured_random_email_monitor.log}"

snapshot() {
  python - "${JOB_ID}" "${OUTPUT_ROOT}" <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

job_id, output_root_text = sys.argv[1:]
output_root = Path(output_root_text)
expected_new = 3712
terminal_states = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"
}

try:
    raw = subprocess.check_output(
        ["sacct", "-j", job_id, "--format=JobIDRaw,State,Elapsed,ExitCode", "-n", "-P"],
        text=True,
        stderr=subprocess.STDOUT,
    )
except Exception as exc:
    raw = f"sacct query failed: {exc}"

rows = []
parent_state = "UNKNOWN"
parent_elapsed = "?"
for line in raw.splitlines():
    fields = line.split("|")
    if len(fields) < 4:
        continue
    row_id, state, elapsed, exit_code = fields[:4]
    if row_id == job_id:
        parent_state, parent_elapsed = state, elapsed
    if row_id.startswith(f"{job_id}_") and "." not in row_id:
        rows.append((row_id, state, elapsed, exit_code))

counts = {}
if output_root.exists():
    for task_dir in output_root.iterdir():
        trajectory_dir = task_dir / "trajectories"
        if trajectory_dir.is_dir():
            counts[task_dir.name] = sum(1 for _ in trajectory_dir.glob("*.json"))

completed = sum(counts.values())
remaining = max(expected_new - completed, 0)
percent = 100.0 * completed / expected_new

elapsed_seconds = 0
parts = parent_elapsed.split("-")
clock = parts[-1].split(":")
try:
    days = int(parts[0]) if len(parts) == 2 else 0
    if len(clock) == 3:
        hours, minutes, seconds = map(int, clock)
    else:
        hours = 0
        minutes, seconds = map(int, clock)
    elapsed_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
except Exception:
    pass

rate_per_min = completed * 60 / elapsed_seconds if elapsed_seconds else 0.0
eta_minutes = remaining / rate_per_min if rate_per_min else None

cost = 0.0
summary_count = 0
for summary_path in output_root.glob("summary*.json") if output_root.exists() else []:
    try:
        payload = json.loads(summary_path.read_text())
        cost += float(payload.get("cost_summary", {}).get("total_cost_usd", 0.0))
        summary_count += 1
    except Exception:
        pass

short = sorted((name, count) for name, count in counts.items() if count < 70)
now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
print(f"RoboCasa tick70 structured-random generation — {now}")
print()
print(f"Slurm array: {job_id} | parent state: {parent_state} | elapsed: {parent_elapsed}")
for row_id, state, elapsed, exit_code in sorted(rows):
    print(f"  {row_id}: {state} elapsed={elapsed} exit={exit_code}")
print()
print(f"Saved new trajectories: {completed}/{expected_new} ({percent:.1f}%)")
print(f"Current throughput: {rate_per_min:.1f} trajectories/min")
print(f"Estimated remaining time: {eta_minutes:.0f} min" if eta_minutes is not None else "Estimated remaining time: not available yet")
print(f"Completed summary cost currently visible: ${cost:.2f} across {summary_count} summary files")
print(f"Task directories visible: {len(counts)}/53")
if short:
    preview = ", ".join(f"{name}={count}" for name, count in short[:15])
    suffix = f" (+{len(short) - 15} more)" if len(short) > 15 else ""
    print(f"Visible tasks below 70 new trajectories: {preview}{suffix}")

states = [state.split()[0] for _, state, _, _ in rows]
all_terminal = len(states) == 5 and all(state in terminal_states for state in states)
print()
print(f"all_terminal={str(all_terminal).lower()}")
PY
}

send_snapshot() {
  local body subject
  body="$(snapshot 2>&1)"
  if grep -q '^all_terminal=true$' <<<"${body}"; then
    subject="${SUBJECT_PREFIX} — final"
  else
    subject="${SUBJECT_PREFIX}"
  fi
  printf '%s\n' "${body}" | mail -s "${subject}" "${EMAIL_TO}"
  {
    printf '\n[%s] emailed %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${EMAIL_TO}"
    printf '%s\n' "${body}"
  } >>"${LOG_FILE}"
  grep -q '^all_terminal=true$' <<<"${body}"
}

while true; do
  if send_snapshot; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
