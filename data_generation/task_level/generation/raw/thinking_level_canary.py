"""Generate one validator-gated canary trajectory at one Gemini thinking level."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from data_generation.task_level.generation.raw.config import RuntimeConfig
from data_generation.task_level.generation.raw.runtime_support import (
    _sampling_strategy_for_runtime,
    _unwrap_generation_response,
    _validate_candidate,
    format_trajectory_variation_key,
)
from data_generation.task_level.runtime.client import build_generation_client
from data_generation.task_level.tasks import get_task_definition


DEFAULT_MODEL = "gemini-3-flash-preview"


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = RuntimeConfig(
        composite_task=args.task,
        num_runs=args.num_runs,
        run_indices=(args.run_index,),
        model=args.model,
        sdk="google-genai",
        project=None,
        location=args.location,
        temperature=args.temperature,
        max_workers=1,
        max_retries=1,
        random_start_location=True,
        random_access_state=True,
        sampling="structured_random",
        thinking_level=args.thinking_level,
        tick_format=True,
        prompt_style="simplified_v3",
        retry_feedback_style="observational",
        partition_policy="none",
    )
    definition = get_task_definition(args.task)
    instance = definition.build_task_instance(args.run_index, config)
    validator = definition.validator_factory(instance)
    strategy = _sampling_strategy_for_runtime(config)
    variation_key = format_trajectory_variation_key(args.run_index, 0)
    prompt = strategy.build_prompt(
        task_definition=definition,
        runtime_config=config,
        task_instance=instance,
        variation_key=variation_key,
    )
    client = build_generation_client(
        sdk=config.sdk,
        project=config.project,
        location=config.location,
        timeout_sec=600,
    )

    started = perf_counter()
    try:
        raw = client.generate(
            model=config.model,
            prompt=prompt,
            response_schema=strategy.response_schema(
                task_definition=definition,
                runtime_config=config,
            ),
            temperature=config.temperature,
            thinking_level=config.thinking_level,
        )
        generation_seconds = perf_counter() - started
        payload, usage = _unwrap_generation_response(raw)
        sampled = strategy.extract_candidates(
            raw_response=payload,
            task_definition=definition,
            runtime_config=config,
            variation_key=variation_key,
        )[0]
        candidate = sampled.candidate
        validation, normalized = _validate_candidate(
            candidate, validator, enforce_validation=False
        )
        result = {
            "task": args.task,
            "run_index": args.run_index,
            "model": config.model,
            "thinking_level": config.thinking_level,
            "temperature": config.temperature,
            "variation_key": variation_key,
            "generation_seconds": generation_seconds,
            "usage": asdict(usage) if usage is not None else None,
            "is_valid": validation.get("is_valid") is True,
            "validation": validation,
            "candidate": normalized if validation.get("is_valid") else candidate,
        }
    except Exception as exc:
        result = {
            "task": args.task,
            "run_index": args.run_index,
            "model": config.model,
            "thinking_level": config.thinking_level,
            "temperature": config.temperature,
            "variation_key": variation_key,
            "generation_seconds": perf_counter() - started,
            "usage": None,
            "is_valid": False,
            "validation": {
                "is_valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            "candidate": None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "task",
                    "run_index",
                    "thinking_level",
                    "generation_seconds",
                    "is_valid",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
