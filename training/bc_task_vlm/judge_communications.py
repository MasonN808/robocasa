"""LLM-as-judge scoring for communicate steps in standalone eval runs.

Exact string match is a near-impossible bar for free-text coordination
messages, so this re-scores every ground-truth communicate step with a judge
model: a prediction counts when it (a) chose the communicate tool, (b)
addressed the same recipient, and (c) conveys the same coordination content
as the expert message (paraphrase allowed; missing or contradicting the
plan's content is a fail).

Reads structured_eval_predictions.jsonl from one or more run dirs, writes
comm_judge.jsonl (per-step verdicts, resumable) and comm_judge_metrics.json:

- comm_judged_match_rate: judged-correct rate over ALL GT communicate steps
  (predicting a non-communicate tool counts as wrong without an API call)
- judged_exact_call_accuracy: exact matches on action steps + judged matches
  on communicate steps, over all steps — the headline "communication-fair"
  accuracy.

Usage:
    python -m training.bc_task_vlm.judge_communications \
        --run-dir training/bc_task_vlm/eval_runs/gemini3flash__heldout_tasks \
        [--run-dir ...] [--judge-model gemini-3-flash-preview]
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from data_generation.task_level.runtime.client import (
    build_raw_google_genai_client,
    load_dotenv_file,
)
from training.bc_task_vlm.evaluation import (
    _is_communicate_record,
    load_prediction_records,
)

JUDGE_SYSTEM = (
    "You judge whether a predicted inter-agent message is equivalent to a "
    "reference message from an expert demonstration of a two-robot kitchen "
    "task. Equivalent means: same recipient, and the message conveys the "
    "same coordination content (who does what next, or the same "
    "acknowledgement). Paraphrasing, different wording, and extra harmless "
    "detail are fine. Missing the key content, naming different objects or "
    "actions, or contradicting the reference is not equivalent."
)

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "equivalent": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["equivalent", "reason"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def judge_one(
    client,
    model: str,
    record: dict[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    from google.genai import types

    target = record["target_tool_call"]
    predicted = record.get("parsed_tool_call")
    target_args = target.get("arguments", {})
    predicted_args = (predicted or {}).get("arguments", {})
    if not target_args:
        raise ValueError(
            f"Empty target arguments for {record['sample_id']}; refusing to "
            "judge vacuous comparisons."
        )
    verdict: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "target_args": target_args,
        "predicted_args": predicted_args,
    }
    if not predicted or predicted.get("name") != "communicate":
        verdict.update(
            equivalent=False,
            reason="prediction is not a communicate call",
            judged_by="rule",
        )
        return verdict

    prompt = (
        f"Reference message (expert):\n{json.dumps(target_args)}\n\n"
        f"Predicted message:\n{json.dumps(predicted_args)}\n\n"
        "Are they equivalent?"
    )
    last_error = None
    for _ in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=JUDGE_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            payload = json.loads(response.text)
            verdict.update(
                equivalent=bool(payload["equivalent"]),
                reason=str(payload.get("reason", "")),
                judged_by=model,
            )
            return verdict
        except Exception as error:  # noqa: BLE001 - retried, then recorded
            last_error = error
    verdict.update(
        equivalent=False,
        reason=f"judge_failed: {last_error}",
        judged_by="error",
    )
    return verdict


def process_run(run_dir: Path, client, args: argparse.Namespace) -> None:
    records = load_prediction_records(run_dir / "structured_eval_predictions.jsonl")
    comm_records = [r for r in records if _is_communicate_record(r)]
    action_records = [r for r in records if not _is_communicate_record(r)]

    judge_path = run_dir / "comm_judge.jsonl"
    current_args = {
        r["sample_id"]: (r.get("parsed_tool_call") or {}).get("arguments", {})
        for r in comm_records
    }
    done: dict[str, dict[str, Any]] = {}
    if judge_path.exists():
        for line in judge_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                verdict = json.loads(line)
                # Error verdicts are retried on resume rather than treated as
                # permanent judgements; empty target_args marks verdicts from
                # the pre-fix judge that compared vacuous payloads.
                if verdict.get("judged_by") == "error" or not verdict.get(
                    "target_args"
                ):
                    continue
                # A verdict is only valid for the prediction it judged. Re-scoring
                # a run (e.g. after a parser fix) changes predictions under stable
                # sample_ids, so a verdict whose predicted_args no longer match is
                # stale and must be re-judged rather than silently reused.
                sample_id = verdict["sample_id"]
                if sample_id in current_args and verdict.get(
                    "predicted_args", {}
                ) != current_args[sample_id]:
                    continue
                done[sample_id] = verdict

    pending = [r for r in comm_records if r["sample_id"] not in done]
    print(f"[{run_dir.name}] {len(comm_records)} comm steps, {len(pending)} to judge")
    if pending:
        with judge_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for verdict in pool.map(
                    lambda r: judge_one(client, args.judge_model, r, args.max_attempts),
                    pending,
                ):
                    handle.write(json.dumps(verdict) + "\n")
                    handle.flush()
                    if verdict["judged_by"] != "error":
                        done[verdict["sample_id"]] = verdict

    # Score strictly over the current prediction records so stale verdicts
    # from an earlier predictions file can never enter the numerator.
    comm_sample_ids = {r["sample_id"] for r in comm_records}
    scored = {sid: v for sid, v in done.items() if sid in comm_sample_ids}
    judge_failures = len(comm_sample_ids) - len(scored)
    comm_judged_matches = sum(bool(v["equivalent"]) for v in scored.values())
    action_exact = sum(bool(r["exact_tool_call_match"]) for r in action_records)

    # Trajectory completion, judged. The exact-match trajectory rate is ~0 for
    # every model because free-text communicate steps (~40% of steps) can
    # essentially never match verbatim; scoring those by judge equivalence
    # instead measures whether a model actually carries a whole episode.
    trajectories: dict[tuple[str, str], list[bool]] = {}
    for record in records:
        if _is_communicate_record(record):
            step_ok = bool(scored.get(record["sample_id"], {}).get("equivalent"))
        else:
            step_ok = bool(record["exact_tool_call_match"])
        trajectories.setdefault(
            (record["task_name"], record["trajectory_id"]), []
        ).append(step_ok)
    traj_full = sum(1 for flags in trajectories.values() if all(flags))
    prefix_fractions = []
    for flags in trajectories.values():
        correct_prefix = 0
        for step_ok in flags:
            if not step_ok:
                break
            correct_prefix += 1
        prefix_fractions.append(correct_prefix / max(len(flags), 1))

    metrics = {
        "comm_step_count": len(comm_records),
        "comm_judged_match_rate": comm_judged_matches / max(len(comm_records), 1),
        "comm_judge_failures": judge_failures,
        "action_exact_call_accuracy": action_exact / max(len(action_records), 1),
        "judged_exact_call_accuracy": (action_exact + comm_judged_matches)
        / max(len(records), 1),
        "judged_trajectory_count": len(trajectories),
        "judged_trajectory_all_steps_rate": traj_full / max(len(trajectories), 1),
        "judged_trajectory_mean_prefix_fraction": (
            sum(prefix_fractions) / max(len(prefix_fractions), 1)
        ),
        "judge_model": args.judge_model,
    }
    (run_dir / "comm_judge_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{run_dir.name}] {json.dumps(metrics, indent=2)}")


def main() -> None:
    args = parse_args()
    load_dotenv_file()
    location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    client = build_raw_google_genai_client(
        args.project,
        location,
        timeout_sec=args.request_timeout,
    )
    for run_dir in args.run_dir:
        process_run(run_dir, client, args)


if __name__ == "__main__":
    main()
