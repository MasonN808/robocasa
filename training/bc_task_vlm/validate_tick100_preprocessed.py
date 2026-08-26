#!/usr/bin/env python3
"""Gate the tick100 no-index artifact before allocating training GPUs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def validate_target_against_displayed_schema(feature: dict) -> None:
    """Require every supervised call to be legal under its own prompt schema."""

    target = feature["target_tool_call"]
    schemas = {
        item["function"]["name"]: item["function"]["parameters"]
        for item in feature["tool_schemas"]
    }
    name = target.get("name")
    assert name in schemas, (feature["sample_id"], name)
    parameters = schemas[name]
    arguments = target.get("arguments")
    assert isinstance(arguments, dict), (feature["sample_id"], arguments)
    properties = parameters.get("properties", {})
    unexpected = set(arguments).difference(properties)
    assert not unexpected, (feature["sample_id"], name, sorted(unexpected))
    missing = set(parameters.get("required", ())).difference(arguments)
    assert not missing, (feature["sample_id"], name, sorted(missing))
    for arg_name, value in arguments.items():
        allowed = properties[arg_name].get("enum")
        if allowed is not None:
            assert value in allowed, (
                feature["sample_id"], name, arg_name, value, allowed
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-train-tasks", type=int, default=47)
    parser.add_argument("--trajectories-per-task", type=int, default=100)
    parser.add_argument("--validation-trajectories-per-task", type=int, default=10)
    args = parser.parse_args()

    config = json.loads((args.artifact / "preprocess_config.json").read_text())
    assert config["sft_format"] == "tool_call"
    assert config["partial_history"] is True
    assert config["partial_step_index_mode"] == "none"
    assert config["partial_observation_mode"] == "consume_once"
    assert config["train_get_image"] is True
    assert len(config["train_tasks"]) == args.expected_train_tasks

    train_ids = config["train_trajectory_ids_by_task"]
    val_ids = config["val_trajectory_ids_by_task"]
    for task in config["train_tasks"]:
        train = set(train_ids[task])
        val = set(val_ids[task])
        assert not train & val, f"split overlap for {task}"
        assert len(train | val) == args.trajectories_per_task, task
        expected_val = args.validation_trajectories_per_task
        expected_train = args.trajectories_per_task - expected_val
        assert len(train) == expected_train and len(val) == expected_val, (
            task,
            len(train),
            len(val),
        )

    dataset = load_from_disk(str(args.artifact / "dataset"))
    assert len(dataset["train"]) > 0 and len(dataset["validation"]) > 0
    assert set(dataset["train"]["task_name"]) == set(config["train_tasks"])
    assert set(dataset["validation"]["task_name"]) == set(config["train_tasks"])

    # This is a full-corpus contract check, not a sample. A target that contains
    # an argument absent from the exact function schema shown in its prompt is
    # malformed SFT data even if the executor or FSM happens to accept it.
    for split_name in ("train", "validation"):
        for row in dataset[split_name]:
            validate_target_against_displayed_schema(json.loads(row["feature_json"]))

    rng = random.Random(42)
    for split_name in ("train", "validation"):
        split = dataset[split_name]
        indices = rng.sample(range(len(split)), min(250, len(split)))
        for index in indices:
            feature = json.loads(split[index]["feature_json"])
            text = json.dumps(feature["messages"], ensure_ascii=False)
            assert " step=" not in text and '"step":' not in text, (
                split_name,
                feature["sample_id"],
            )
            for image_path in feature["image_paths"]:
                resolved_image = Path(image_path)
                if not resolved_image.is_absolute():
                    resolved_image = args.artifact / resolved_image
                assert resolved_image.is_file(), resolved_image
            target = feature["target_tool_call"]
            assert isinstance(target.get("name"), str) and isinstance(
                target.get("arguments"), dict
            )

    print(
        f"validated tick100 artifact: train_rows={len(dataset['train'])} "
        f"validation_rows={len(dataset['validation'])} tasks={len(config['train_tasks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
