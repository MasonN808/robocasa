#!/usr/bin/env bash
set -uo pipefail

EMAIL_TO="${EMAIL_TO:-dbenhamougol@umass.edu}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-RoboCasa no-index OOTB eval progress}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/training/bc_task_vlm/eval_runs/ots_noindex_email_monitor.log}"

PILOT_IDS=(299404 299405 299406 299407 309014 309015)
RUN_SPECS=(
  "299411|Gemini 3 Flash / held-out tasks|90|training/bc_task_vlm/eval_runs/livesim_ots-noindex-gemini3flash__heldout_tasks/live_sim_trajectories.jsonl"
  "299412|Gemini 3 Flash / held-out trajectories|141|training/bc_task_vlm/eval_runs/livesim_ots-noindex-gemini3flash__heldout_trajectories/live_sim_trajectories.jsonl"
  "299413|Gemini Robotics ER2 / held-out tasks|90|training/bc_task_vlm/eval_runs/livesim_ots-noindex-er2__heldout_tasks/live_sim_trajectories.jsonl"
  "299414|Gemini Robotics ER2 / held-out trajectories|141|training/bc_task_vlm/eval_runs/livesim_ots-noindex-er2__heldout_trajectories/live_sim_trajectories.jsonl"
  "309018|Qwen3-VL-8B base / held-out tasks|90|training/bc_task_vlm/eval_runs/livesim_ots-noindex-qwen3vl8b__heldout_tasks/live_sim_trajectories.jsonl"
  "309019|Qwen3-VL-8B base / held-out trajectories|141|training/bc_task_vlm/eval_runs/livesim_ots-noindex-qwen3vl8b__heldout_trajectories/live_sim_trajectories.jsonl"
)

snapshot() {
  cd "${REPO_ROOT}"
  python - "${PILOT_IDS[*]}" "${RUN_SPECS[@]}" <<'PY'
import json
import os
import subprocess
import sys
import time
from collections import Counter

pilot_ids = sys.argv[1].split()
specs = [arg.split("|", 3) for arg in sys.argv[2:]]
full_ids = [spec[0] for spec in specs]
all_ids = pilot_ids + full_ids
try:
    raw = subprocess.check_output(
        ["sacct", "-j", ",".join(all_ids), "--format=JobIDRaw,State,Elapsed,ExitCode", "-n", "-P"],
        text=True,
    )
except Exception as exc:
    raw = f"sacct query failed: {exc}"

states = {}
for line in raw.splitlines():
    fields = line.split("|")
    if len(fields) >= 4 and fields[0] in all_ids:
        states[fields[0]] = (fields[1], fields[2], fields[3])

print(time.strftime("RoboCasa out-of-the-box no-index evals — %Y-%m-%d %H:%M:%S %Z"))
print()
print("Pilot gates:")
for job_id in pilot_ids:
    state, elapsed, exit_code = states.get(job_id, ("UNKNOWN", "?", "?"))
    print(f"  {job_id}: {state} {elapsed} exit={exit_code}")

print("\nFull evaluations:")
terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
all_terminal = True
now = time.time()
for job_id, label, expected_text, path in specs:
    expected = int(expected_text)
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
    all_terminal &= state.split()[0] in terminal_states
    fsm = sum(bool(row.get("fsm_goal_satisfied")) for row in rows)
    native = sum(bool(row.get("native_success")) for row in rows)
    terms = Counter(row.get("termination", "missing") for row in rows)
    print(
        f"  {job_id} {label}: {len(rows)}/{expected} | {state} {elapsed} exit={exit_code} | "
        f"latest age {age} | FSM/native {fsm}/{native} | JSON errors {parse_errors} | "
        f"terminations {dict(terms)}"
    )
print(f"\nall_full_terminal={str(all_terminal).lower()}")
PY
}

send_snapshot() {
  local body subject
  body="$(snapshot 2>&1)"
  if grep -q '^all_full_terminal=true$' <<<"${body}"; then
    subject="${SUBJECT_PREFIX} — final"
    if [[ "$(grep -c '| COMPLETED ' <<<"${body}")" == "${#RUN_SPECS[@]}" ]]; then
      local package_output
      package_output="$({
        cd "${REPO_ROOT}"
        python training/bc_task_vlm/build_corrected_eval_results_artifact.py
        cd /home/dbenhamougol_umass/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
        PATH=/tmp/node-v22.17.0-linux-x64/bin:${PATH} npm run report:deliver -- \
          --input "${REPO_ROOT}/training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/artifact.json" \
          --output "${REPO_ROOT}/training/bc_task_vlm/eval_runs/corrected_index_ab_results_artifact/report.html"
      } 2>&1)" || true
      body+=$'\n\nArtifact refresh:\n'
      body+="${package_output}"
    else
      body+=$'\n\nArtifact refresh skipped because one or more full jobs did not complete successfully.'
    fi
  else
    subject="${SUBJECT_PREFIX}"
  fi
  printf '%s\n' "${body}" | mail -s "${subject}" "${EMAIL_TO}"
  {
    printf '\n[%s] emailed %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${EMAIL_TO}"
    printf '%s\n' "${body}"
  } >>"${LOG_FILE}"
  grep -q '^all_full_terminal=true$' <<<"${body}"
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
