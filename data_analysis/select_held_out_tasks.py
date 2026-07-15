#!/usr/bin/env python3
"""Select held-out tasks for the task-VLM SFT-necessity experiment.

Profiles every verified task from its spec and a sample of raw generated
trajectories (tool set, tool frequencies, tool-order n-grams, argument
patterns), then greedily picks the most order/argument-novel tasks whose tool
sets stay fully covered by the remaining train tasks.

Usage:
    python -m data_analysis.select_held_out_tasks \
        --raw-root data_generation/task_level/data/raw/20260430T030150Z_full_run \
        --local-image-root data_generation/task_level/data/image/20260430T030150Z_full_run \
        --hf-repo-list training/bc_task_vlm/hf_repo_lists/robocasa_20260430T030150Z_full_49.txt \
        --output-dir data_analysis/analysis/held_out_task_selection
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from data_generation.task_level.tasks.specs import load_verified_task_specs
from training.bc_task_vlm.task_registry import _camel_to_snake_case

MESSAGE_LIKE_ARGS = {"message"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--local-image-root", type=Path, required=True)
    parser.add_argument("--hf-repo-list", type=Path, required=True)
    parser.add_argument("--num-held-out", type=int, default=5)
    parser.add_argument("--trajectories-per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-train-tasks-per-tool",
        type=int,
        default=2,
        help="Every tool used by a held-out task must be used by at least this many train tasks.",
    )
    parser.add_argument(
        "--max-tool-set-jaccard",
        type=float,
        default=0.8,
        help="Reject a candidate whose tool set is near-identical to an already selected one.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def hf_task_names(hf_repo_list: Path) -> set[str]:
    names: set[str] = set()
    for line in hf_repo_list.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(line.split("_full_run_", 1)[1])
    return names


def sample_trajectory_paths(task_raw_dir: Path, limit: int, seed: int) -> list[Path]:
    paths = sorted((task_raw_dir / "trajectories").glob("traj_*.json"))
    if len(paths) <= limit:
        return paths
    sampler = random.Random(seed)
    return sorted(sampler.sample(paths, limit))


def profile_task(
    *,
    task_name: str,
    spec_tools: dict[str, dict[str, Any]],
    trajectory_paths: list[Path],
) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    trigram_counts: Counter[tuple[str, str, str]] = Counter()
    arg_signatures: Counter[tuple[str, tuple[str, ...]]] = Counter()
    symbolic_vocab: Counter[str] = Counter()
    lengths: list[int] = []

    for path in trajectory_paths:
        trajectory = json.loads(path.read_text())
        tools = [step["tool"] for step in trajectory["steps"]]
        lengths.append(len(tools))
        tool_counts.update(tools)
        bigram_counts.update(zip(tools, tools[1:]))
        trigram_counts.update(zip(tools, tools[1:], tools[2:]))
        for step in trajectory["steps"]:
            args = step.get("args", {}) or {}
            arg_signatures[(step["tool"], tuple(sorted(args)))] += 1
            for key, value in args.items():
                if key in MESSAGE_LIKE_ARGS:
                    continue
                if isinstance(value, str):
                    symbolic_vocab[value] += 1
                elif isinstance(value, list):
                    symbolic_vocab.update(v for v in value if isinstance(v, str))

    return {
        "task_name": task_name,
        "spec_tool_set": sorted(spec_tools),
        "data_tool_set": sorted(tool_counts),
        "num_trajectories_profiled": len(trajectory_paths),
        "avg_trajectory_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "tool_counts": dict(tool_counts),
        "bigram_counts": {" -> ".join(k): v for k, v in bigram_counts.items()},
        "trigram_counts": {" -> ".join(k): v for k, v in trigram_counts.items()},
        "arg_signatures": {
            f"{tool}({', '.join(keys)})": count
            for (tool, keys), count in arg_signatures.items()
        },
        "symbolic_vocab": dict(symbolic_vocab),
    }


def _distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _jensen_shannon(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        result = 0.0
        for key in keys:
            pa = a.get(key, 0.0)
            if pa <= 0.0:
                continue
            result += pa * math.log2(pa / b[key])
        return result

    mixture = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in keys}
    return 0.5 * _kl(p, mixture) + 0.5 * _kl(q, mixture)


def _unseen_mass(own_counts: dict[str, int], other_types: set[str]) -> float:
    total = sum(own_counts.values())
    if total == 0:
        return 0.0
    unseen = sum(count for key, count in own_counts.items() if key not in other_types)
    return unseen / total


def score_task(profile: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, float]:
    pooled_tools: Counter[str] = Counter()
    other_bigrams: set[str] = set()
    other_trigrams: set[str] = set()
    other_arg_signatures: set[str] = set()
    other_vocab: set[str] = set()
    for other in others:
        pooled_tools.update(other["tool_counts"])
        other_bigrams.update(other["bigram_counts"])
        other_trigrams.update(other["trigram_counts"])
        other_arg_signatures.update(other["arg_signatures"])
        other_vocab.update(other["symbolic_vocab"])

    tool_jsd = _jensen_shannon(
        _distribution(profile["tool_counts"]), _distribution(dict(pooled_tools))
    )
    unseen_bigram = _unseen_mass(profile["bigram_counts"], other_bigrams)
    unseen_trigram = _unseen_mass(profile["trigram_counts"], other_trigrams)
    unseen_args = _unseen_mass(profile["arg_signatures"], other_arg_signatures)
    unseen_vocab = _unseen_mass(profile["symbolic_vocab"], other_vocab)

    novelty = (
        0.30 * tool_jsd
        + 0.15 * unseen_bigram
        + 0.25 * unseen_trigram
        + 0.15 * unseen_args
        + 0.15 * unseen_vocab
    )
    return {
        "tool_frequency_jsd": tool_jsd,
        "unseen_bigram_mass": unseen_bigram,
        "unseen_trigram_mass": unseen_trigram,
        "unseen_arg_signature_mass": unseen_args,
        "unseen_symbolic_vocab_mass": unseen_vocab,
        "novelty": novelty,
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def select_held_out(
    profiles: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, float]],
    *,
    num_held_out: int,
    min_train_tasks_per_tool: int,
    max_tool_set_jaccard: float,
) -> tuple[list[str], dict[str, str]]:
    """Greedy pick by novelty subject to tool coverage + diversity constraints."""

    selected: list[str] = []
    rejections: dict[str, str] = {}
    ranked = sorted(profiles, key=lambda name: scores[name]["novelty"], reverse=True)
    for candidate in ranked:
        if len(selected) == num_held_out:
            break
        train_tasks = [
            name for name in profiles if name != candidate and name not in selected
        ]
        tool_support = Counter()
        for name in train_tasks:
            tool_support.update(set(profiles[name]["data_tool_set"]))
        uncovered = [
            tool
            for tool in profiles[candidate]["data_tool_set"]
            if tool_support[tool] < min_train_tasks_per_tool
        ]
        if uncovered:
            rejections[candidate] = (
                f"tools not covered by >= {min_train_tasks_per_tool} train tasks: {uncovered}"
            )
            continue
        candidate_tools = set(profiles[candidate]["data_tool_set"])
        near_duplicate = next(
            (
                picked
                for picked in selected
                if _jaccard(candidate_tools, set(profiles[picked]["data_tool_set"]))
                > max_tool_set_jaccard
            ),
            None,
        )
        if near_duplicate is not None:
            rejections[candidate] = f"tool set too similar to already selected {near_duplicate}"
            continue
        selected.append(candidate)
    return selected, rejections


def write_markdown(
    output_path: Path,
    *,
    selected: list[str],
    profiles: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, float]],
    rejections: dict[str, str],
) -> None:
    lines = [
        "# Held-out task selection",
        "",
        "Tasks ranked by novelty of tool usage/order/arguments relative to the remaining",
        "train pool; every held-out tool stays covered by the train tasks.",
        "",
        "| Task | Novelty | Tool JSD | Unseen 3-gram | Unseen args | Unseen vocab | Tools |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in selected:
        score = scores[name]
        tools = ", ".join(
            tool for tool in profiles[name]["data_tool_set"] if tool != "communicate"
        )
        lines.append(
            f"| {name} | {score['novelty']:.3f} | {score['tool_frequency_jsd']:.3f} "
            f"| {score['unseen_trigram_mass']:.3f} | {score['unseen_arg_signature_mass']:.3f} "
            f"| {score['unseen_symbolic_vocab_mass']:.3f} | {tools} |"
        )
    if rejections:
        lines += ["", "## Higher-novelty candidates rejected", ""]
        for name, reason in rejections.items():
            lines.append(f"- `{name}`: {reason}")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hf_names = hf_task_names(args.hf_repo_list)
    profiles: dict[str, dict[str, Any]] = {}
    availability: dict[str, dict[str, bool]] = {}
    for spec in load_verified_task_specs():
        task_name = _camel_to_snake_case(spec.composite_task)
        on_hf = task_name in hf_names
        local = (args.local_image_root / task_name).is_dir()
        availability[task_name] = {"hf": on_hf, "local_rendered": local}
        if not (on_hf or local):
            print(f"[skip] {task_name}: no HF repo and no local rendered data")
            continue
        raw_dir = args.raw_root / task_name
        if not raw_dir.is_dir():
            print(f"[skip] {task_name}: no raw trajectories at {raw_dir}")
            continue
        paths = sample_trajectory_paths(raw_dir, args.trajectories_per_task, args.seed)
        profiles[task_name] = profile_task(
            task_name=task_name,
            spec_tools=spec.allowed_tool_specs,
            trajectory_paths=paths,
        )
        print(
            f"[profiled] {task_name}: {len(paths)} traj, "
            f"tools={len(profiles[task_name]['data_tool_set'])}, "
            f"avg_len={profiles[task_name]['avg_trajectory_length']:.1f}"
        )

    scores = {
        name: score_task(profile, [p for other, p in profiles.items() if other != name])
        for name, profile in profiles.items()
    }
    selected, rejections = select_held_out(
        profiles,
        scores,
        num_held_out=args.num_held_out,
        min_train_tasks_per_tool=args.min_train_tasks_per_tool,
        max_tool_set_jaccard=args.max_tool_set_jaccard,
    )
    train_tasks = sorted(name for name in profiles if name not in selected)

    result = {
        "held_out_tasks": selected,
        "train_tasks": train_tasks,
        "rejections": rejections,
        "availability": availability,
        "scores": scores,
        "selection_config": {
            "num_held_out": args.num_held_out,
            "trajectories_per_task": args.trajectories_per_task,
            "seed": args.seed,
            "min_train_tasks_per_tool": args.min_train_tasks_per_tool,
            "max_tool_set_jaccard": args.max_tool_set_jaccard,
            "raw_root": str(args.raw_root),
        },
    }
    (args.output_dir / "held_out_task_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "task_profiles.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(
        args.output_dir / "held_out_task_selection.md",
        selected=selected,
        profiles=profiles,
        scores=scores,
        rejections=rejections,
    )
    print(f"\nHeld-out ({len(selected)}): {', '.join(selected)}")
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Wrote {args.output_dir}/held_out_task_selection.json")


if __name__ == "__main__":
    main()
