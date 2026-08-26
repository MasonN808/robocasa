"""Gate production on demo-generation and exact SFT/live prompt contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data_generation.task_level.generation.raw.cascade_canary import FLASH_MODEL, _config
from data_generation.task_level.generation.raw.runtime_support import (
    _sampling_strategy_for_runtime,
    format_trajectory_variation_key,
)
from data_generation.task_level.subatomic_tool_specs import build_model_tool_specs
from data_generation.task_level.tasks import get_task_definition
from training.bc_task_vlm.prompting import (
    PROMPT_CONTRACT_VERSION,
    build_partial_user_prompt,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _representative_configurations(manifest: dict) -> list[dict]:
    """Return one production configuration per task across manifest versions."""

    if isinstance(manifest.get("tasks"), list):
        return [
            {
                "task": entry["task"],
                "run_index": int(entry["configurations"][0]["run_index"]),
                "num_runs": int(entry["configurations"][0]["num_runs"]),
            }
            for entry in manifest["tasks"]
        ]
    configurations = manifest.get("configurations")
    if not isinstance(configurations, dict):
        raise ValueError("Unsupported prompt-gate manifest schema")
    by_task: dict[str, dict] = {}
    for split in configurations.values():
        if not isinstance(split, dict):
            continue
        for entries in split.values():
            if not entries:
                continue
            entry = entries[0]
            task = entry["composite_task"]
            by_task.setdefault(
                task,
                {
                    "task": task,
                    # The frozen cohort stores physical configurations rather
                    # than generator run indices.  Rank is a stable valid seed
                    # for rendering representative production prompts.
                    "run_index": int(entry.get("configuration_rank", 0)),
                    "num_runs": max(1, len(entries)),
                },
            )
    return [by_task[task] for task in sorted(by_task)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    failures = []
    model_tools = build_model_tool_specs(include_get_image=True)
    for task_entry in _representative_configurations(manifest):
        task = task_entry["task"]
        run_index = int(task_entry["run_index"])
        namespace = argparse.Namespace(
            task=task,
            num_runs=int(task_entry["num_runs"]),
            run_index=run_index,
            location="global",
            temperature=0.6,
        )
        config = _config(namespace, model=FLASH_MODEL, thinking="low")
        definition = get_task_definition(task)
        instance = definition.build_task_instance(run_index, config)
        strategy = _sampling_strategy_for_runtime(config)
        generation_prompt = strategy.build_prompt(
            task_definition=definition,
            runtime_config=config,
            task_instance=instance,
            variation_key=format_trajectory_variation_key(run_index, 0),
        )
        common = dict(
            composite_task=task,
            task_instruction=instance.task_goal or "",
            agent_id="agent_0",
            history_steps=[],
            observation_views=[],
            allowed_tool_specs=model_tools,
            partial_step_index_mode="none",
            global_step_index=0,
            coordinator_id=instance.coordinator_id,
            initial_state=instance.initial_state,
        )
        sft_prompt = build_partial_user_prompt(**common)
        eval_prompt = build_partial_user_prompt(**common)
        checks = {
            "sft_eval_byte_identical": sft_prompt == eval_prompt,
            "no_step_index": "Next global step index" not in sft_prompt
            and "Next local agent turn index" not in sft_prompt,
            "generation_simplified_v3_tick": "simultaneous ticks" in generation_prompt,
            "generation_has_initial_state": "Initial task state:" in generation_prompt,
            "generation_has_coordinator": str(instance.coordinator_id) in generation_prompt,
        }
        if not all(checks.values()):
            failures.append({"task": task, "checks": checks})
        rows.append(
            {
                "task": task,
                "run_index": run_index,
                "coordinator_id": instance.coordinator_id,
                "generation_prompt_sha256": _sha(generation_prompt),
                "sft_prompt_sha256": _sha(sft_prompt),
                "eval_prompt_sha256": _sha(eval_prompt),
                "checks": checks,
            }
        )
    report = {
        "valid": not failures,
        "task_count": len(rows),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "generation_prompt_style": "simplified_v3",
        "generation_tick_format": True,
        "partial_step_index_mode": "none",
        "failures": failures,
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("valid", "task_count", "prompt_contract_version", "generation_prompt_style")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
