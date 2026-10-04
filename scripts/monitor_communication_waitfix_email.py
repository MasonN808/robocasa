"""Email array progress every 30 minutes; stop after a final terminal report."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "training/bc_task_vlm/reports/communication_waitfix_v3/email_monitor.log"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def snapshot(array_id):
    result = subprocess.run(
        ["sacct", "-j", array_id, "--format=JobID%40,State%40,Elapsed,ExitCode", "-n", "-P"],
        text=True, capture_output=True, check=True, timeout=60,
    )
    states = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4:
            states[fields[0]] = fields[1:4]
    queue = subprocess.run(
        ["squeue", "-r", "-j", array_id, "-h", "-o", "%i|%T|%M"],
        text=True, capture_output=True, timeout=60,
    )
    for line in queue.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 3:
            states[fields[0]] = [fields[1], fields[2], "not exited"]
    lines = [f"RoboTalk communication reruns — {datetime.now().astimezone().isoformat()}",
             f"Array {array_id}: Qwen Instruct and Gemini Flash; unguided, minimal, intermediate.", ""]
    done = True
    for index in range(12):
        job = f"{array_id}_{index}"
        model = "Qwen Instruct" if index < 6 else "Gemini Flash"
        mode = ("unguided", "minimal", "intermediate")[(index % 6) // 2]
        split = "trained tasks (470 episodes)" if index % 2 == 0 else "held-out tasks (60 episodes)"
        state, elapsed, code = states.get(job, ["UNKNOWN", "?", "?"])
        done = done and state.split()[0].rstrip("+") in TERMINAL
        lines.append(f"{job}: {model}, {mode}, {split}: {state}; elapsed {elapsed}; exit {code}")
    lines += ["", "All jobs terminal; monitor stopping (terminal does not necessarily mean successful)."
              if done else "Next update in 30 minutes. Missing accounting records are not treated as completion."]
    return "\n".join(lines), done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--array-id", required=True, type=str)
    args = parser.parse_args()
    if not args.array_id.isdigit():
        parser.error("--array-id must be a numeric Slurm array ID")
    while True:
        done = False
        try:
            body, done = snapshot(args.array_id)
        except Exception as exc:
            body = f"RoboTalk monitor: status query failed; will retry in 30 minutes.\n{exc}"
        if args.dry_run:
            print(body)
            return
        subject = "RoboTalk communication evals — " + ("final status" if done else "30-minute update")
        try:
            subprocess.run(["mail", "-s", subject, "dbenhamougol@umass.edu"],
                           input=body + "\n", text=True, check=True, timeout=60)
            status = "Email accepted by local mail command"
        except Exception as exc:
            status = f"Email submission failed: {exc}"
            done = False  # Retry even a final email if delivery submission failed.
        with LOG.open("a") as handle:
            handle.write(f"\n{datetime.now().astimezone().isoformat()} {status}\n{body}\n")
        if done:
            return
        time.sleep(1800)


if __name__ == "__main__":
    main()
