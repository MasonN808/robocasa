"""Prompt generation for low-level tool-call probes."""

from __future__ import annotations


def _label(value) -> str:
    return str(value).replace("_", " ")


def prompt_for_tool_call(step: dict) -> str:
    tool = step.get("tool")
    args = step.get("args") or {}

    if tool == "navigate_to_fixture":
        return f"navigate to the {_label(args.get('fixture_id', 'target fixture'))}"
    if tool == "give_space":
        return f"move away from the {_label(args.get('fixture_id', 'target fixture'))}"
    if tool == "open_hinged_part":
        return f"open the {_label(args.get('target_id', 'target fixture'))}"
    if tool == "close_hinged_part":
        return f"close the {_label(args.get('target_id', 'target fixture'))}"
    if tool == "open_sliding_part":
        return f"open the {_label(args.get('target_id', 'drawer'))}"
    if tool == "close_sliding_part":
        return f"close the {_label(args.get('target_id', 'drawer'))}"
    if tool == "pick_up_object":
        obj = _label(args.get("object_id", "object"))
        source = args.get("source_id")
        if source:
            return f"pick up the {obj} from the {_label(source)}"
        return f"pick up the {obj}"
    if tool == "place_in_receptacle":
        return f"place the {_label(args.get('object_id', 'object'))} in the {_label(args.get('receptacle_id') or args.get('target_id') or 'receptacle')}"
    if tool == "place_on_object":
        return f"place the {_label(args.get('object_id', 'object'))} on the {_label(args.get('support_object_id') or args.get('target_id') or 'support object')}"
    if tool == "place_on_surface":
        return f"place the {_label(args.get('object_id', 'object'))} on the {_label(args.get('support_id') or args.get('target_id') or 'surface')}"
    if tool == "place_next_to":
        ref = args.get("reference_object_id") or args.get("reference_id") or args.get("reference_fixture_id") or "reference"
        return f"place the {_label(args.get('object_id', 'object'))} next to the {_label(ref)}"
    if tool == "place_under":
        return f"place the {_label(args.get('object_id', 'object'))} under the {_label(args.get('reference_fixture_id') or args.get('target_id') or 'fixture')}"
    if tool == "press_button":
        return f"press the {_label(args.get('control_id', 'button'))} on the {_label(args.get('target_id', 'fixture'))}"
    if tool == "press_lever":
        return f"press the {_label(args.get('control_id', 'lever'))} on the {_label(args.get('target_id', 'fixture'))}"
    if tool == "set_rotary_control":
        return f"set the {_label(args.get('control_id', 'control'))} on the {_label(args.get('target_id', 'fixture'))} to {_label(args.get('goal', 'the target state'))}"
    return str(step.get("reasoning") or tool or "perform the requested action")
