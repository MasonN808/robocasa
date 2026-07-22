"""Instrument an HF live-sim evaluation with component-level timing events.

Pass live_sim_eval arguments after ``--``. Evaluation outputs remain governed by
the normal evaluator and should stay under the git-ignored eval_runs directory.
"""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import training.bc_task_vlm.live_sim_eval as live
from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


EVENTS_PATH: Path | None = None

events: dict[str, list[dict[str, Any]]] = {
    "generation": [],
    "inference_pass": [],
    "proposal_turn": [],
    "render": [],
    "fsm_step": [],
    "adapt_step": [],
    "sim_execute": [],
    "trajectory_load": [],
    "native_check": [],
}
active_pass: dict[str, Any] | None = None
current_trajectory_id: str | None = None
pass_counts: dict[tuple[str, int], int] = {}


def _write_events() -> None:
    if EVENTS_PATH is None:
        return
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "events": events,
        }
        EVENTS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    except Exception:
        traceback.print_exc()


atexit.register(_write_events)


original_generate = live.HfPolicy.generate


def instrumented_generate(self, feature: dict[str, Any]) -> str:
    """Equivalent to HfPolicy.generate, with synchronized component timings."""

    torch = self._torch
    row: dict[str, Any] = {
        "sample_id": feature.get("sample_id"),
        "trajectory_id": feature.get("trajectory_id"),
        "step_index": feature.get("step_index"),
        "num_images": len(feature.get("image_paths") or []),
        **(active_pass or {}),
    }
    total_started = time.perf_counter()
    try:
        started = time.perf_counter()
        batch = self.collator([feature])
        batch.pop("sample_metadata", None)
        row["collate_s"] = time.perf_counter() - started
        row["prompt_tokens"] = int(batch["input_ids"].shape[1])

        device = next(self.model.parameters()).device
        started = time.perf_counter()
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        row["host_to_device_s"] = time.perf_counter() - started

        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        row["model_generate_s"] = time.perf_counter() - started

        trimmed = generated[0][batch["input_ids"].shape[1] :]
        row["output_tokens"] = int(trimmed.shape[0])
        started = time.perf_counter()
        decoded = self.processor.decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        row["decode_s"] = time.perf_counter() - started
        row["decoded_chars"] = len(decoded)
        if torch.cuda.is_available():
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        return decoded
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row["total_s"] = time.perf_counter() - total_started
        events["generation"].append(row)


live.HfPolicy.generate = instrumented_generate


original_generate_once = live._generate_once


def instrumented_generate_once(*args, **kwargs):
    global active_pass

    trajectory_id = str((kwargs.get("trajectory") or {}).get("trajectory_id") or "")
    step_index = int(kwargs.get("step_index", -1))
    key = (trajectory_id, step_index)
    ordinal = pass_counts.get(key, 0) + 1
    pass_counts[key] = ordinal
    context = {
        "trajectory_id": trajectory_id,
        "step_index": step_index,
        "pass_ordinal": ordinal,
        "views": list(kwargs.get("views") or []),
    }
    previous = active_pass
    active_pass = context
    started = time.perf_counter()
    row = dict(context)
    try:
        result = original_generate_once(*args, **kwargs)
        row["result_tool"] = result.get("tool")
        row["result_error"] = result.get("error")
        return result
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["inference_pass"].append(row)
        active_pass = previous


live._generate_once = instrumented_generate_once


original_model_propose = live._model_propose_step


def instrumented_model_propose(*args, **kwargs):
    started = time.perf_counter()
    row = {
        "trajectory_id": str((kwargs.get("trajectory") or {}).get("trajectory_id") or ""),
        "step_index": int(kwargs.get("step_index", -1)),
    }
    try:
        proposal, views = original_model_propose(*args, **kwargs)
        row["final_tool"] = proposal.get("tool")
        row["final_error"] = proposal.get("error")
        row["final_views"] = list(views or [])
        return proposal, views
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["proposal_turn"].append(row)


live._model_propose_step = instrumented_model_propose


original_render_views = live.SimSession.render_views


def instrumented_render_views(self, views, *, agent_id, out_dir, tag):
    started = time.perf_counter()
    row = {
        "trajectory_id": current_trajectory_id,
        "step_index": (active_pass or {}).get("step_index"),
        "pass_ordinal": (active_pass or {}).get("pass_ordinal"),
        "views": list(views),
        "num_views": len(views),
        "agent_id": agent_id,
    }
    try:
        return original_render_views(
            self,
            views,
            agent_id=agent_id,
            out_dir=out_dir,
            tag=tag,
        )
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["render"].append(row)


live.SimSession.render_views = instrumented_render_views


original_fsm_step = live.FsmMirror.step


def instrumented_fsm_step(self, step):
    started = time.perf_counter()
    row = {
        "trajectory_id": current_trajectory_id,
        "step_index": step.get("step"),
        "tool": step.get("tool"),
    }
    try:
        legal, reason = original_fsm_step(self, step)
        row["legal"] = legal
        row["reason"] = reason
        return legal, reason
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["fsm_step"].append(row)


live.FsmMirror.step = instrumented_fsm_step


original_adapt_step = TrajectoryAdapter._adapt_step


def instrumented_adapt_step(self, step, *args, **kwargs):
    started = time.perf_counter()
    row = {
        "trajectory_id": current_trajectory_id,
        "step_index": step.get("step"),
        "tool": step.get("tool"),
    }
    try:
        return original_adapt_step(self, step, *args, **kwargs)
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["adapt_step"].append(row)


TrajectoryAdapter._adapt_step = instrumented_adapt_step


original_execute = SimToolExecutor.execute


def instrumented_execute(self, tool_name, *args, **kwargs):
    started = time.perf_counter()
    row = {
        "trajectory_id": current_trajectory_id,
        "tool": tool_name,
    }
    try:
        result = original_execute(self, tool_name, *args, **kwargs)
        row["success"] = bool(result.success)
        return result
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["sim_execute"].append(row)


SimToolExecutor.execute = instrumented_execute


original_start_trajectory = live.SimSession.start_trajectory


def instrumented_start_trajectory(self, trajectory):
    global current_trajectory_id

    current_trajectory_id = str(trajectory.get("trajectory_id") or "")
    started = time.perf_counter()
    row = {"trajectory_id": current_trajectory_id}
    try:
        return original_start_trajectory(self, trajectory)
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["trajectory_load"].append(row)


live.SimSession.start_trajectory = instrumented_start_trajectory


original_native_success = live.SimSession.native_success


def instrumented_native_success(self):
    started = time.perf_counter()
    row = {"trajectory_id": current_trajectory_id}
    try:
        success, error = original_native_success(self)
        row["success"] = success
        row["error"] = error
        return success, error
    finally:
        row["elapsed_s"] = time.perf_counter() - started
        events["native_check"].append(row)


live.SimSession.native_success = instrumented_native_success


def _forwarded_option(args: list[str], name: str) -> str | None:
    try:
        index = args.index(name)
    except ValueError:
        return None
    if index + 1 >= len(args):
        return None
    return args[index + 1]


def main() -> None:
    global EVENTS_PATH

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass all live_sim_eval arguments after --.",
    )
    parser.add_argument(
        "--timing-events-path",
        type=Path,
        default=None,
        help="Timing JSON path (default: <output-dir>/timing_events.json).",
    )
    parser.add_argument("live_args", nargs=argparse.REMAINDER)
    wrapper_args = parser.parse_args()
    live_args = list(wrapper_args.live_args)
    if live_args[:1] == ["--"]:
        live_args = live_args[1:]
    if not live_args:
        parser.error("live_sim_eval arguments are required after --")
    output_dir = _forwarded_option(live_args, "--output-dir")
    if output_dir is None:
        parser.error("forwarded arguments must include --output-dir")
    EVENTS_PATH = (
        wrapper_args.timing_events_path
        or Path(output_dir) / "timing_events.json"
    )
    sys.argv = ["live_sim_timing_benchmark.py", *live_args]
    try:
        live.main()
    finally:
        _write_events()


if __name__ == "__main__":
    main()
