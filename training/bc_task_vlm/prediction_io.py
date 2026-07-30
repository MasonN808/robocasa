"""Dependency-light readers for `structured_eval_predictions.jsonl`.

Deliberately imports nothing beyond the standard library. `evaluation.py`
pulls in torch/transformers at module scope, so anything that only needs to
*read back* prediction records — notably `judge_communications.py`, which
just calls a hosted judge model — would otherwise drag the whole training
stack into its environment. Keeping these helpers here lets the judge run in
a minimal venv holding only the Google GenAI client.

`evaluation.py` re-exports both names, so existing
`from training.bc_task_vlm.evaluation import load_prediction_records`
imports keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prediction_records(predictions_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _is_communicate_record(record: dict[str, Any]) -> bool:
    return (record.get("target_tool_call") or {}).get("name") == "communicate"
