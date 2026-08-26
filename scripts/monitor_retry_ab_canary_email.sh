#!/usr/bin/env bash
set -uo pipefail

EMAIL_TO="${EMAIL_TO:-dbenhamougol@umass.edu}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-RoboCasa 53x2 retry A/B canary}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/slurm_logs/retry_ab_canary_email_monitor.log}"
TARGETED_ROOT="${TARGETED_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick53x2_retry_targeted_atomic_v1}"
OBSERVATIONAL_ROOT="${OBSERVATIONAL_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick53x2_retry_observational_atomic_v1}"
JOB_IDS=(329567 329568)

snapshot() {
  python - "${JOB_IDS[0]}" "${TARGETED_ROOT}" targeted \
    "${JOB_IDS[1]}" "${OBSERVATIONAL_ROOT}" observational <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

specs = [sys.argv[index:index + 3] for index in range(1, len(sys.argv), 3)]
job_ids = [job_id for job_id, _, _ in specs]
terminal_states = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"
}
try:
    raw = subprocess.check_output(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State,Elapsed,ExitCode", "-n", "-P"],
        text=True,
        stderr=subprocess.STDOUT,
    )
except Exception as exc:
    raw = f"sacct query failed: {exc}"

rows_by_job = {job_id: [] for job_id in job_ids}
for line in raw.splitlines():
    fields = line.split("|")
    if len(fields) < 4 or "." in fields[0]:
        continue
    for job_id in job_ids:
        if fields[0].startswith(f"{job_id}_"):
            rows_by_job[job_id].append(tuple(fields[:4]))

print(time.strftime("RoboCasa 53x2 retry A/B canary — %Y-%m-%d %H:%M:%S %Z"))
print()
all_terminal = True
for job_id, root_text, label in specs:
    root = Path(root_text)
    summaries = []
    for path in root.glob("*/summary.json") if root.exists() else ():
        try:
            summaries.append(json.loads(path.read_text()))
        except Exception:
            pass
    accepted = sum(int(item.get("num_trajectories", 0)) for item in summaries)
    requested = sum(int(item.get("num_runs", 0)) for item in summaries)
    states = [row[1].split()[0] for row in rows_by_job[job_id]]
    terminal = len(states) == 5 and all(state in terminal_states for state in states)
    all_terminal = all_terminal and terminal
    state_counts = {state: states.count(state) for state in sorted(set(states))}
    print(
        f"{label} job {job_id}: shards={state_counts} | "
        f"task summaries={len(summaries)}/53 | accepted={accepted}/{requested or 106}"
    )
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

if [[ "${1:-}" == "--once" ]]; then
  send_snapshot || true
  exit 0
fi

while true; do
  if send_snapshot; then
    exit 0
  fi
  sleep "${INTERVAL_SECONDS}"
done
