#!/usr/bin/env bash
set -uo pipefail

EMAIL_TO="${EMAIL_TO:-dbenhamougol@umass.edu}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1200}"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-RoboCasa schema-fixed SFT progress}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/slurm_logs/schemafix_pipeline_email_monitor.log}"

JOB_IDS=(334118 334119 334120 334121 334122 334127 334128 334129 334130 334131 334146)

snapshot() {
  cd "${REPO_ROOT}"
  python - <<'PY'
import json, os, re, subprocess, time
from pathlib import Path

jobs = {
    "334118": "27/3 training", "334119": "27/3 trajectory pilot",
    "334120": "27/3 task pilot", "334121": "27/3 trajectory full",
    "334122": "27/3 task full", "334127": "90/10 training",
    "334128": "90/10 trajectory pilot", "334129": "90/10 task pilot",
    "334130": "90/10 trajectory full", "334131": "90/10 task full",
    "334146": "final artifact",
}
raw = subprocess.check_output([
    "sacct", "-j", ",".join(jobs),
    "--format=JobIDRaw,State,Elapsed,ExitCode", "-n", "-P"
], text=True)
states = {}
for line in raw.splitlines():
    f = line.split("|")
    if len(f) >= 4 and f[0] in jobs:
        states[f[0]] = (f[1], f[2], f[3])

print(time.strftime("RoboCasa schema-fixed pipeline — %Y-%m-%d %H:%M:%S %Z"))
print("\nJobs:")
for jid, label in jobs.items():
    state, elapsed, exit_code = states.get(jid, ("UNKNOWN", "?", "?"))
    print(f"  {jid} {label}: {state} elapsed={elapsed} exit={exit_code}")

print("\nTraining steps:")
for jid, total in (("334118", 2166), ("334127", 7215)):
    path = Path(f"/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/v3-matrix-{jid}.err")
    text = path.read_text(errors="replace")[-250000:] if path.exists() else ""
    matches = re.findall(r"(\d+)/(\d+).*?<([^,]+),", text.replace("\r", "\n"))
    if not matches:
        pairs = re.findall(r"(\d+)/(" + str(total) + r")", text.replace("\r", "\n"))
        print(f"  {jid}: {pairs[-1][0]}/{total}" if pairs else f"  {jid}: no progress line yet")
    else:
        step, parsed_total, eta = matches[-1]
        print(f"  {jid}: {step}/{parsed_total}, displayed ETA {eta}")

print("\nCorrected out-of-box rollouts:")
root = Path("training/bc_task_vlm/eval_runs")
for source, traj_n, task_n in (("tick30_revalidated_v3", 141, 90), ("tick100_from150_v3", 470, 60)):
    for model in ("qwen3vl8b", "gemini3flash", "er2"):
        counts = []
        for split, expected in (("heldout_trajectories", traj_n), ("heldout_tasks", task_n)):
            path = root / f"livesim_ots-noindex-{model}-{source}__{split}-schemafix/live_sim_trajectories.jsonl"
            rows = 0
            successes = 0
            if path.exists():
                for line in path.read_text().splitlines():
                    if not line.strip(): continue
                    rows += 1
                    try: successes += bool(json.loads(line).get("fsm_goal_satisfied"))
                    except Exception: pass
            counts.append(f"{split.replace('heldout_', '')} {rows}/{expected} FSM={successes}")
        print(f"  {source} / {model}: " + "; ".join(counts))

artifact_state = states.get("334146", ("UNKNOWN", "?", "?"))[0].split()[0]
print(f"\nall_done={str(artifact_state in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'}).lower()}")
PY
}

send_snapshot() {
  local body subject
  body="$(snapshot 2>&1)"
  subject="${SUBJECT_PREFIX}"
  grep -q '^all_done=true$' <<<"${body}" && subject="${SUBJECT_PREFIX} — final"
  printf '%s\n' "${body}" | mail -s "${subject}" "${EMAIL_TO}"
  {
    printf '\n[%s] emailed %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${EMAIL_TO}"
    printf '%s\n' "${body}"
  } >>"${LOG_FILE}"
  grep -q '^all_done=true$' <<<"${body}"
}

while true; do
  if send_snapshot; then exit 0; fi
  sleep "${INTERVAL_SECONDS}"
done
