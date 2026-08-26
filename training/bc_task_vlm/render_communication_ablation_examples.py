#!/usr/bin/env python3
"""Replay and caption three representative Gemini 3 communication-ablation traces."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import textwrap

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data_generation.task_level.tasks.shared.instances import initial_state_from_configuration
from data_generation.task_level.tasks.specs import load_verified_task_specs
from training.bc_task_vlm.live_sim_eval import SimSession


EVAL_ROOT = ROOT / "training/bc_task_vlm/eval_runs/fixed_live_sim"
SELECTIONS = (
    {
        "group": "full_comm_both_agents_success",
        "slug": "01_setup_soda_bowl",
        "run": "ootb_gemini3flash_fixed10_promptv9",
        "id": "train_task_types:setup_soda_bowl:sampled:20260817:006:d91a73af630b:r0",
        "title": "Full communication - both agents succeed",
        "subtitle": "Setup soda bowl | Error-free coordination with an explicit fridge handoff.",
    },
    {
        "group": "full_comm_both_agents_success", "slug": "02_candle_cleanup",
        "run": "ootb_gemini3flash_fixed10_promptv9",
        "id": "train_task_types:candle_cleanup:sampled:20260817:004:4e99c7185217:r0",
        "title": "Full communication - both agents succeed",
        "subtitle": "Candle cleanup | Error-free success with balanced physical contributions.",
    },
    {
        "group": "full_comm_both_agents_success", "slug": "03_spicy_marinade",
        "run": "ootb_gemini3flash_fixed10_promptv9",
        "id": "train_task_types:spicy_marinade:sampled:20260817:007:ff3e9f629b72:r1",
        "title": "Full communication - both agents succeed",
        "subtitle": "Spicy marinade | Error-free coordinated success by both agents.",
    },
    {
        "group": "full_comm_both_agents_success", "slug": "04_alcohol_serving_prep",
        "run": "ootb_gemini3flash_fixed10_promptv9",
        "id": "train_task_types:alcohol_serving_prep:sampled:20260817:006:fc98cca8e3f2:r0",
        "title": "Full communication - both agents succeed",
        "subtitle": "Alcohol serving prep | Error-free coordinated success by both agents.",
    },
    {
        "group": "full_comm_both_agents_success", "slug": "05_gather_marinade_ingredients",
        "run": "ootb_gemini3flash_fixed10_promptv9",
        "id": "train_task_types:gather_marinade_ingredients:sampled:20260817:001:fe318ee15122:r0",
        "title": "Full communication - both agents succeed",
        "subtitle": "Gather marinade ingredients | Error-free coordinated success by both agents.",
    },
    {
        "group": "no_comm_single_agent_success", "slug": "01_setup_soda_bowl",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:setup_soda_bowl:sampled:20260817:009:653732a29a62:r0",
        "title": "No communication - single-agent success",
        "subtitle": "Agent 0 transfers both sodas; Agent 1 never performs a successful task action.",
    },
    {
        "group": "no_comm_single_agent_success", "slug": "02_display_meat_variety",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:display_meat_variety:sampled:20260817:004:d8e08a570905:r0",
        "title": "No communication - single-agent success",
        "subtitle": "Agent 1 completes every successful physical task action.",
    },
    {
        "group": "no_comm_single_agent_success", "slug": "03_prepare_sausage_cheese",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:prepare_sausage_cheese:sampled:20260817:009:551c0fbe188e:r0",
        "title": "No communication - single-agent success",
        "subtitle": "Agent 1 completes every successful physical task action.",
    },
    {
        "group": "no_comm_single_agent_success", "slug": "04_add_lemon_to_fish",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:add_lemon_to_fish:sampled:20260817:005:9aa08beedc3d:r0",
        "title": "No communication - single-agent success",
        "subtitle": "Agent 1 completes the task while Agent 0 performs no successful physical action.",
    },
    {
        "group": "no_comm_single_agent_success", "slug": "05_lemon_seasoning_fish",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:lemon_seasoning_fish:sampled:20260817:001:13bbc7dfaf3b:r0",
        "title": "No communication - single-agent success",
        "subtitle": "Agent 0 completes the task while Agent 1 performs no successful physical action.",
    },
    {
        "group": "no_comm_coordination_failure", "slug": "01_setup_soda_bowl",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:setup_soda_bowl:sampled:20260817:000:da6a51a231d0:r0",
        "title": "No communication - coordination failure",
        "subtitle": "Repeated fridge conflicts create a livelock until the budget expires.",
    },
    {
        "group": "no_comm_coordination_failure", "slug": "02_display_meat_variety",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:display_meat_variety:sampled:20260817:008:dd3c32cdc15b:r0",
        "title": "No communication - coordination failure",
        "subtitle": "Repeated fridge contention prevents completion.",
    },
    {
        "group": "no_comm_coordination_failure", "slug": "03_prepare_sausage_cheese",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:prepare_sausage_cheese:sampled:20260817:008:9d7a3098eaf9:r0",
        "title": "No communication - coordination failure",
        "subtitle": "Fixture contention leads to budget exhaustion.",
    },
    {
        "group": "no_comm_coordination_failure", "slug": "04_plate_store_dinner",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:plate_store_dinner:sampled:20260817:005:9d7a3098eaf9:r0",
        "title": "No communication - coordination failure",
        "subtitle": "Both agents remain active, but repeated conflicts prevent completion.",
    },
    {
        "group": "no_comm_coordination_failure", "slug": "05_distribute_chicken",
        "run": "ootb_gemini3flash_fixed10_promptv9_comm_none_v2",
        "id": "train_task_types:distribute_chicken:sampled:20260817:002:33678d2b8926:r0",
        "title": "No communication - coordination failure",
        "subtitle": "Both agents act, but uncoordinated contention leaves the goal unsatisfied.",
    },
)


def read_record(run: str, trajectory_id: str) -> dict:
    path = EVAL_ROOT / run / "train_task_types/aggregate/live_sim_trajectories.jsonl"
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("trajectory_id") == trajectory_id:
                return row
    raise KeyError(trajectory_id)


def read_episode(run: str, record: dict) -> dict:
    manifests = EVAL_ROOT / run / "train_task_types/manifests"
    signature = record["configuration_signature"]
    for path in sorted(manifests.glob("worker_*.json")):
        data = json.loads(path.read_text())
        episodes = data["configurations"]["train_task_types"].get(record["task_name"], [])
        for episode in episodes:
            if episode["configuration_signature"] == signature:
                return episode
    raise KeyError(signature)


def make_trajectory(record: dict, episode: dict) -> dict:
    specs = {spec.composite_task: spec for spec in load_verified_task_specs()}
    spec = specs[record["composite_task"]]
    return {
        "trajectory_id": record["trajectory_id"],
        "task": spec.task_goal,
        "composite_task": record["composite_task"],
        "initial_state": initial_state_from_configuration(spec.initial_state, episode["configuration"]),
        "grounding_map": deepcopy(spec.grounding),
        "coordinator_id": record.get("coordinator_id"),
        "steps": [],
    }


def font(size: int, *, bold: bool = False):
    candidates = [
        f"/usr/share/fonts/urw-base35/NimbusSansNarrow-{'Bold' if bold else 'Regular'}.otf",
        f"/usr/share/fonts/google-droid-sans-fonts/DroidSans{'-Bold' if bold else ''}.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_image(array, size):
    im = Image.fromarray(array).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (10, 13, 18))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def wrap_to_pixels(draw: ImageDraw.ImageDraw, value: str, text_font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def panel_action(step: dict | None, panel_agent: str) -> tuple[str, str, tuple[int, int, int]]:
    if step is None:
        return "READY", "Initial state", (92, 180, 255)
    proposal = step["proposal"]
    if proposal.get("agent", step.get("agent")) != panel_agent:
        return "NO ACTION", "This step", (142, 151, 166)
    tool = str(proposal.get("tool", "unknown"))
    args = proposal.get("args", {})
    if step.get("conflict"):
        heading, color = "BLOCKED", (255, 101, 101)
    elif not step.get("legal", True) or step.get("sim_success") is False:
        heading, color = "FAILED", (255, 150, 85)
    elif tool == "communicate":
        heading = f"MESSAGE TO {args.get('to', '?').upper()}"
        if args.get("releases"):
            release_value = args["releases"]
            if isinstance(release_value, list):
                release_value = ", ".join(str(value) for value in release_value)
            heading += f"  |  RELEASES: {str(release_value).upper()}"
        if args.get("coordination_phase"):
            phase = str(args["coordination_phase"]).replace("_", " ").upper()
            heading += f"  |  PHASE: {phase}"
        color = (105, 210, 255)
    elif tool in {"wait_for_signal", "wait"}:
        heading, color = "WAITING", (235, 194, 83)
    elif tool == "get_image":
        heading, color = "OBSERVING", (172, 160, 255)
    else:
        heading, color = "ACTING", (92, 214, 138)
    detail = tool.replace("_", " ").upper()
    if tool == "communicate":
        detail = f"\"{args.get('message', '')}\""
    elif tool in {"wait_for_signal", "wait"}:
        source = args.get("from", "partner")
        about = args.get("about")
        detail = f"WAITING FOR SIGNAL FROM {source}"
        if about:
            detail += f" ABOUT {about}"
    else:
        target = next((args[key] for key in (
            "object_id", "fixture_id", "target_id", "support_object_id", "source_id"
        ) if key in args), None)
        if target is not None:
            detail += f" | {str(target).replace('_', ' ')}"
    return heading, detail, color


def render_card(images: dict, selection: dict, record: dict, step: dict | None, ordinal: int,
                total: int, panel_states: dict[str, dict | None] | None = None):
    canvas = Image.new("RGB", (1920, 1080), (11, 15, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 18), selection["title"], font=font(40, bold=True), fill=(248, 249, 252))
    goal_text = f"TASK GOAL: {selection['task_goal']}"
    draw.text((34, 70), goal_text, font=font(23, bold=True), fill=(188, 198, 214))

    panels = [("top_view", "TOP VIEW"), ("robot0_agentview_center", "AGENT 0"),
              ("robot1_agentview_center", "AGENT 1")]
    panel_specs = ((20, 400), (440, 710), (1170, 710))
    for (x, width), (key, label) in zip(panel_specs, panels):
        array = images.get(key)
        if array is None:
            array = images.get("top_view")
        if array is not None:
            canvas.paste(fit_image(array, (width, 940)), (x, 120))
        draw.rounded_rectangle((x, 120, x + width, 1060), radius=10, outline=(68, 80, 99), width=2)
        label_width = draw.textbbox((0, 0), label, font=font(23, bold=True))[2]
        draw.rounded_rectangle((x + 14, 134, x + 36 + label_width, 176), radius=8, fill=(11, 15, 23))
        draw.text((x + 25, 140), label, font=font(23, bold=True), fill=(245, 247, 250))
        if key.startswith("robot"):
            panel_agent = "agent_0" if key.startswith("robot0") else "agent_1"
            panel_step = (panel_states or {}).get(panel_agent)
            if panel_step is None:
                continue
            heading, detail, badge_color = panel_action(panel_step, panel_agent)
            panel_tool = (((panel_step or {}).get("proposal") or {}).get("tool"))
            is_message = panel_tool == "communicate"
            is_idle = heading in {"NO ACTION", "READY"}
            heading_font = 28
            detail_font = 32
            detail_face = font(detail_font, bold=True)
            detail_lines = [] if is_idle else wrap_to_pixels(draw, detail, detail_face, width - 68)
            line_height = 36
            card_top = 875
            draw.rounded_rectangle((x + 14, card_top, x + width - 14, 1040), radius=12,
                                   fill=(11, 15, 23), outline=badge_color, width=3)
            heading_y = card_top + 15
            heading_face = font(heading_font, bold=True)
            while heading_font > 20 and draw.textbbox((0, 0), heading, font=heading_face)[2] > width - 68:
                heading_font -= 2
                heading_face = font(heading_font, bold=True)
            draw.text((x + 34, heading_y), heading, font=heading_face, fill=badge_color)
            if not is_idle:
                detail_y = heading_y + 40
                for detail_line in detail_lines:
                    draw.text((x + 34, detail_y), detail_line, font=detail_face,
                              fill=(245, 247, 250))
                    detail_y += line_height

    if step is None:
        status = "INITIAL STATE"
        lines = [f"Task: {record['task_name']}", f"Gemini 3 | communication={record['communication_mode']}"]
        color = (92, 180, 255)
    else:
        proposal = step["proposal"]
        tool, args = proposal["tool"], proposal.get("args", {})
        agent = proposal.get("agent", step.get("agent", "?"))
        if step.get("conflict"):
            status, color = "BLOCKED - RESOURCE CONFLICT", (255, 101, 101)
        elif not step.get("legal", True) or step.get("sim_success") is False:
            status, color = "ACTION FAILED", (255, 150, 85)
        elif tool == "communicate":
            status, color = "MESSAGE DELIVERED", (105, 210, 255)
        elif tool in {"wait_for_signal", "wait"}:
            status, color = "WAITING FOR PARTNER", (235, 194, 83)
        elif tool == "get_image":
            status, color = "OBSERVING", (172, 160, 255)
        else:
            status, color = "ACTION EXECUTED", (92, 214, 138)
        lines = [f"Tick {ordinal}/{total} | t={step.get('sim_time', 0):g}s"]
        if tool == "communicate":
            lines.append(f"To {args.get('to')}: \"{args.get('message', '')}\"")
        elif args:
            lines.append(", ".join(f"{k}={v}" for k, v in args.items()))
        if step.get("reason"):
            lines.append(str(step["reason"]))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/communication_ablation_gemini3_examples")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--render-size", type=int, default=700)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for selection in SELECTIONS:
        record = read_record(selection["run"], selection["id"])
        episode = read_episode(selection["run"], record)
        trajectory = make_trajectory(record, episode)
        render_selection = {**selection, "task_goal": trajectory["task"]}
        scene = record["scene"]
        session = SimSession(composite_task=record["composite_task"], sample_trajectory=trajectory,
                             layout=scene["layout"], style=scene["style"], seed=scene["seed"],
                             gl_backend="egl", render_size=args.render_size, map_dpi=60, map_renderer="raster")
        try:
            adapter, adapted = session.start_trajectory(trajectory)
            group_dir = args.output_dir / selection["group"]
            group_dir.mkdir(parents=True, exist_ok=True)
            output = group_dir / f"{selection['slug']}.mp4"
            ticks: list[list[dict]] = []
            for trace_step in record["steps"]:
                if not ticks or ticks[-1][0].get("sim_time") != trace_step.get("sim_time"):
                    ticks.append([])
                ticks[-1].append(trace_step)
            with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=None) as writer:
                panel_states = {"agent_0": None, "agent_1": None}
                initial = render_card(session.executor.render(), render_selection, record, None, 0,
                                      len(ticks), panel_states)
                for _ in range(args.fps * 2):
                    writer.append_data(np.asarray(initial))
                for ordinal, tick_steps in enumerate(ticks, 1):
                    for panel_agent, previous in list(panel_states.items()):
                        previous_tool = ((previous or {}).get("proposal") or {}).get("tool")
                        if previous_tool not in {"wait_for_signal", "wait"}:
                            panel_states[panel_agent] = None
                    chosen: dict[str, dict] = {}
                    for step in tick_steps:
                        proposal = step["proposal"]
                        tool = proposal.get("tool")
                        acting_agent = proposal.get("agent", step.get("agent"))
                        current = chosen.get(acting_agent)
                        if current is None or current["proposal"].get("tool") == "get_image":
                            chosen[acting_agent] = step
                        if step.get("executed") and step.get("legal", True) and step.get("sim_success") is not False \
                                and tool not in {"get_image", "communicate", "wait_for_signal", "wait"}:
                            symbolic = {"step": step.get("step_index"), "agent": acting_agent,
                                        "tool": tool, "args": proposal.get("args", {})}
                            call = adapter._adapt_step(symbolic, resolved_initial_state=adapted["initial_state"], output_dir=None)
                            session.executor.execute(call["tool"], robot_idx=call["robot_idx"], **call["args"])
                    panel_states.update(chosen)
                    representative = next((s for s in tick_steps if s.get("conflict")), tick_steps[0])
                    card = render_card(session.executor.render(), render_selection, record, representative, ordinal,
                                       len(ticks), panel_states)
                    repeats = args.fps * 3 if any(
                        s["proposal"].get("tool") == "communicate" for s in tick_steps
                    ) else args.fps * 2
                    for _ in range(repeats):
                        writer.append_data(np.asarray(card))
                final = render_card(session.executor.render(), render_selection, record, record["steps"][-1],
                                    len(ticks), len(ticks), panel_states)
                for _ in range(args.fps * 2):
                    writer.append_data(np.asarray(final))
            metadata.append({**selection, "output": str(output), "task": record["task_name"],
                             "fsm_success": record["fsm_goal_satisfied"], "native_success": record["native_success"],
                             "partial_goal_fraction": record["partial_goal_fraction"],
                             "resource_conflicts": record["resource_conflicts"], "steps": len(record["steps"])})
            print(output, flush=True)
        finally:
            session.close()
    (args.output_dir / "selection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
