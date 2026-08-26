#!/usr/bin/env bash
set -uo pipefail

EMAIL_TO="${EMAIL_TO:-dbenhamougol@umass.edu}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-RoboCasa eval progress}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/training/bc_task_vlm/eval_runs/indexab_email_monitor.log}"

JOB_IDS=(290438 290439 290440 290441)
RUN_SPECS=(
  "290438|none/tasks corrected|90|training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxnone-noreason-indexab-savefix-8h__heldout_tasks-bounded-v3-contextfix/live_sim_trajectories.jsonl"
  "290439|none/trajectories corrected|141|training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxnone-noreason-indexab-savefix-8h__heldout_trajectories-bounded-v3-contextfix/live_sim_trajectories.jsonl"
  "290440|local/tasks corrected|90|training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxlocal-noreason-indexab-savefix-8h__heldout_tasks-bounded-v3-contextfix/live_sim_trajectories.jsonl"
  "290441|local/trajectories corrected|141|training/bc_task_vlm/eval_runs/livesim_qwen3vl-8b-v3-tick30-partial-idxlocal-noreason-indexab-savefix-8h__heldout_trajectories-bounded-v3-contextfix/live_sim_trajectories.jsonl"
)

snapshot() {
  cd "${REPO_ROOT}"
  python - "${RUN_SPECS[@]}" <<'PY'
import json
import os
import subprocess
import sys
import time
from collections import Counter

specs = [arg.split("|", 3) for arg in sys.argv[1:]]
job_ids = [spec[0] for spec in specs]
try:
    raw = subprocess.check_output(
        ["sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State,Elapsed,ExitCode", "-n", "-P"],
        text=True,
    )
except Exception as exc:
    raw = f"sacct query failed: {exc}"

states = {}
for line in raw.splitlines():
    fields = line.split("|")
    if len(fields) >= 4 and fields[0] in job_ids:
        states[fields[0]] = (fields[1], fields[2], fields[3])

now = time.time()
print(time.strftime("RoboCasa corrected index A/B eval progress — %Y-%m-%d %H:%M:%S %Z"))
print()
all_terminal = True
for job_id, label, total_text, path in specs:
    total = int(total_text)
    rows = []
    parse_errors = 0
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    parse_errors += 1
        age = f"{now - os.path.getmtime(path):.0f}s"
    else:
        age = "missing"
    state, elapsed, exit_code = states.get(job_id, ("UNKNOWN", "?", "?"))
    terminal = state.split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    all_terminal = all_terminal and terminal
    terms = Counter(row.get("termination", "missing") for row in rows)
    fsm = sum(bool(row.get("fsm_goal_satisfied")) for row in rows)
    native = sum(bool(row.get("native_success")) for row in rows)
    errors = sum(row.get("native_error") is not None for row in rows)
    print(
        f"{job_id} {label}: {len(rows)}/{total} | {state} {elapsed} | "
        f"latest age {age} | FSM/native {fsm}/{native} | native errors {errors} | "
        f"JSON errors {parse_errors} | terminations {dict(terms)}"
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
