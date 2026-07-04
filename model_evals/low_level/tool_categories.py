"""Tool categories for low-level trajectory probing."""

from __future__ import annotations

PHYSICAL_TOOLS = frozenset(
    {
        "navigate_to_fixture",
        "give_space",
        "pick_up_object",
        "place_in_receptacle",
        "place_on_object",
        "place_next_to",
        "place_on_surface",
        "place_under",
        "open_hinged_part",
        "close_hinged_part",
        "open_sliding_part",
        "close_sliding_part",
        "press_button",
        "press_lever",
        "set_rotary_control",
    }
)

NON_PHYSICAL_TOOLS = frozenset({"communicate", "wait", "get_image"})

PRIMARY_TOOLS = frozenset(
    {
        "navigate_to_fixture",
        "pick_up_object",
        "place_in_receptacle",
        "place_on_object",
        "place_next_to",
        "place_on_surface",
        "open_hinged_part",
    }
)

SECONDARY_TOOLS = frozenset({"give_space", "open_sliding_part", "close_hinged_part", "close_sliding_part"})
RARE_TOOLS = PHYSICAL_TOOLS - PRIMARY_TOOLS - SECONDARY_TOOLS


def support_tier(tool_name: str) -> str:
    if tool_name in PRIMARY_TOOLS:
        return "primary"
    if tool_name in SECONDARY_TOOLS:
        return "secondary"
    if tool_name in RARE_TOOLS:
        return "rare"
    return "non_physical"


def expand_tool_filter(tools: list[str] | None) -> set[str]:
    if not tools or "all_physical" in tools or "all" in tools:
        return set(PHYSICAL_TOOLS)
    expanded: set[str] = set()
    for tool in tools:
        if tool == "primary":
            expanded.update(PRIMARY_TOOLS)
        elif tool == "secondary":
            expanded.update(SECONDARY_TOOLS)
        elif tool == "rare":
            expanded.update(RARE_TOOLS)
        elif tool == "physical":
            expanded.update(PHYSICAL_TOOLS)
        else:
            expanded.add(tool)
    return expanded
