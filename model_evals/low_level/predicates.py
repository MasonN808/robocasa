"""Physical success predicates for low-level VLA probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import robocasa.utils.object_utils as OU


@dataclass
class PredicateResult:
    success: bool
    metrics: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "metrics": self.metrics, "reason": self.reason}


def _fail(reason: str, **metrics) -> PredicateResult:
    return PredicateResult(False, metrics, reason)


def _success(reason: str, **metrics) -> PredicateResult:
    return PredicateResult(True, metrics, reason)


def _robot_xy(executor, robot_idx: int) -> np.ndarray:
    return np.asarray(executor.runner._get_robot_position(robot_idx)[:2], dtype=float)


def _fixture_xy(executor, fixture_id: str) -> np.ndarray:
    fixture = executor.runner._fixtures[fixture_id]
    return np.asarray(fixture.pos[:2], dtype=float)


def _object_pose(executor, object_id: str) -> tuple[np.ndarray, np.ndarray]:
    return executor._get_object_pose(object_id)


def _object_xy(executor, object_id: str) -> np.ndarray:
    pos, _ = _object_pose(executor, object_id)
    return np.asarray(pos[:2], dtype=float)


def _object_z(executor, object_id: str) -> float:
    pos, _ = _object_pose(executor, object_id)
    return float(pos[2])


def _joint_value(executor, target_id: str, part_or_control_id: str) -> float | None:
    fixture = executor.runner._fixtures.get(target_id)
    if fixture is None:
        return None
    try:
        joint_name = executor._resolve_joint_name(fixture, part_or_control_id)
    except Exception:
        if part_or_control_id == "sliding":
            try:
                joint_name = executor._resolve_joint_name(fixture, "slide")
            except Exception:
                return None
        else:
            return None
    return executor._joint_qpos_scalar(joint_name)


def _near_fixture_result(executor, fixture_id: str, robot_idx: int, threshold: float = 0.85) -> PredicateResult:
    if fixture_id not in executor.runner._fixtures:
        return _fail(f"unknown fixture {fixture_id!r}")
    dist = float(np.linalg.norm(_robot_xy(executor, robot_idx) - _fixture_xy(executor, fixture_id)))
    if dist <= threshold:
        return _success("robot base is near target fixture", distance=dist, threshold=threshold)
    return _fail("robot base is not near target fixture", distance=dist, threshold=threshold)


def navigate_to_fixture(executor, args: dict, robot_idx: int) -> PredicateResult:
    return _near_fixture_result(executor, args["fixture_id"], robot_idx)


def give_space(executor, args: dict, robot_idx: int) -> PredicateResult:
    fixture_id = args["fixture_id"]
    dist = float(np.linalg.norm(_robot_xy(executor, robot_idx) - _fixture_xy(executor, fixture_id)))
    threshold = 1.2
    if dist >= threshold:
        return _success("robot base moved away from fixture", distance=dist, threshold=threshold)
    return _fail("robot base is still close to fixture", distance=dist, threshold=threshold)


def pick_up_object(executor, args: dict, robot_idx: int) -> PredicateResult:
    object_id = args["object_id"]
    holder = executor._held_by_robot(object_id)
    if holder == robot_idx:
        return _success("object is held by target robot", holder=holder)
    obj_xy = _object_xy(executor, object_id)
    robot_xy = _robot_xy(executor, robot_idx)
    dist = float(np.linalg.norm(obj_xy - robot_xy))
    z = _object_z(executor, object_id)
    if dist <= 0.35 and z >= 0.75:
        return _success("object is near robot and lifted", distance=dist, object_z=z)
    return _fail("object is not held/lifted near robot", holder=holder, distance=dist, object_z=z)


def _object_near_target(executor, object_id: str, target_xy: np.ndarray, threshold: float, reason: str) -> PredicateResult:
    obj_xy = _object_xy(executor, object_id)
    dist = float(np.linalg.norm(obj_xy - np.asarray(target_xy, dtype=float)))
    if dist <= threshold:
        return _success(reason, distance=dist, threshold=threshold)
    return _fail(f"not {reason}", distance=dist, threshold=threshold)


def place_in_receptacle(executor, args: dict, robot_idx: int) -> PredicateResult:
    target_id = args.get("receptacle_id") or args.get("target_id")
    object_id = args["object_id"]
    holder = executor._held_by_robot(object_id)

    if target_id in executor.env.objects:
        obj_pos, _ = _object_pose(executor, object_id)
        target_pos, _ = _object_pose(executor, target_id)
        xy_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        try:
            contact = bool(
                executor.env.check_contact(
                    executor.env.objects[object_id],
                    executor.env.objects[target_id],
                )
            )
            in_receptacle = bool(OU.check_obj_in_receptacle(executor.env, object_id, target_id))
        except Exception as exc:
            return _fail(
                f"object receptacle check failed: {exc}",
                holder=holder,
                xy_distance=xy_dist,
            )
        if holder is None and in_receptacle:
            return _success(
                "object is in receptacle object",
                holder=holder,
                contact=contact,
                xy_distance=xy_dist,
            )
        return _fail(
            "object is not in receptacle object",
            holder=holder,
            contact=contact,
            in_receptacle=in_receptacle,
            xy_distance=xy_dist,
        )

    obj_pos, _ = _object_pose(executor, object_id)
    target_xy = _fixture_xy(executor, target_id)
    xy_dist = float(np.linalg.norm(obj_pos[:2] - target_xy))
    try:
        inside = bool(OU.obj_inside_of(executor.env, object_id, target_id, partial_check=True, th=0.02))
    except Exception as exc:
        return _fail(
            f"fixture receptacle check failed: {exc}",
            holder=holder,
            xy_distance=xy_dist,
        )
    if holder is None and inside:
        return _success(
            "object is inside receptacle fixture",
            holder=holder,
            inside=inside,
            xy_distance=xy_dist,
        )
    return _fail(
        "object is not inside receptacle fixture",
        holder=holder,
        inside=inside,
        xy_distance=xy_dist,
    )


def place_on_object(executor, args: dict, robot_idx: int) -> PredicateResult:
    object_id = args["object_id"]
    support_id = args.get("support_object_id") or args.get("target_id")
    obj_pos, _ = _object_pose(executor, object_id)
    support_pos, _ = _object_pose(executor, support_id)
    xy_dist = float(np.linalg.norm(obj_pos[:2] - support_pos[:2]))
    dz = float(obj_pos[2] - support_pos[2])
    if xy_dist <= 0.18 and dz >= 0.0:
        return _success("object is on/near support object", xy_distance=xy_dist, z_delta=dz)
    return _fail("object is not on support object", xy_distance=xy_dist, z_delta=dz)


def place_on_surface(executor, args: dict, robot_idx: int) -> PredicateResult:
    target_id = args.get("support_id") or args.get("target_id")
    return _object_near_target(executor, args["object_id"], _fixture_xy(executor, target_id), 0.55, "object is on/near target surface")


def place_next_to(executor, args: dict, robot_idx: int) -> PredicateResult:
    ref_id = args.get("reference_object_id") or args.get("reference_id")
    if ref_id and ref_id in executor.env.objects:
        target_xy = _object_xy(executor, ref_id)
        threshold = 0.28
    else:
        ref_id = args.get("reference_fixture_id") or args.get("reference_id")
        target_xy = _fixture_xy(executor, ref_id)
        threshold = 0.55
    return _object_near_target(executor, args["object_id"], target_xy, threshold, "object is next to reference")


def place_under(executor, args: dict, robot_idx: int) -> PredicateResult:
    ref_id = args.get("reference_fixture_id") or args.get("target_id")
    return _object_near_target(executor, args["object_id"], _fixture_xy(executor, ref_id), 0.45, "object is under/near reference fixture")


def open_hinged_part(executor, args: dict, robot_idx: int) -> PredicateResult:
    value = _joint_value(executor, args["target_id"], args["part_id"])
    if value is None:
        return _near_fixture_result(executor, args["target_id"], robot_idx)
    threshold = 0.35
    if value >= threshold:
        return _success("hinged part joint is open", joint_qpos=value, threshold=threshold)
    return _fail("hinged part joint is not open", joint_qpos=value, threshold=threshold)


def close_hinged_part(executor, args: dict, robot_idx: int) -> PredicateResult:
    value = _joint_value(executor, args["target_id"], args["part_id"])
    if value is None:
        return _near_fixture_result(executor, args["target_id"], robot_idx)
    threshold = 0.15
    if value <= threshold:
        return _success("hinged part joint is closed", joint_qpos=value, threshold=threshold)
    return _fail("hinged part joint is not closed", joint_qpos=value, threshold=threshold)


def open_sliding_part(executor, args: dict, robot_idx: int) -> PredicateResult:
    value = _joint_value(executor, args["target_id"], args["part_id"])
    threshold = 0.15
    if value is not None and abs(float(value)) >= threshold:
        return _success("sliding part joint is open", joint_qpos=value, abs_joint_qpos=abs(float(value)), threshold=threshold)
    return _fail("sliding part joint is not open", joint_qpos=value, abs_joint_qpos=abs(float(value)) if value is not None else None, threshold=threshold)


def close_sliding_part(executor, args: dict, robot_idx: int) -> PredicateResult:
    value = _joint_value(executor, args["target_id"], args["part_id"])
    threshold = 0.05
    if value is not None and abs(float(value)) <= threshold:
        return _success("sliding part joint is closed", joint_qpos=value, abs_joint_qpos=abs(float(value)), threshold=threshold)
    return _fail("sliding part joint is not closed", joint_qpos=value, abs_joint_qpos=abs(float(value)) if value is not None else None, threshold=threshold)


def press_button(executor, args: dict, robot_idx: int) -> PredicateResult:
    fixture = executor.runner._fixtures.get(args["target_id"])
    turned_on = bool(getattr(fixture, "_turned_on", False))
    value = _joint_value(executor, args["target_id"], args["control_id"])
    if turned_on or (value is not None and value >= 0.5):
        return _success("button/control is activated", turned_on=turned_on, joint_qpos=value)
    return _fail("button/control is not activated", turned_on=turned_on, joint_qpos=value)


def press_lever(executor, args: dict, robot_idx: int) -> PredicateResult:
    return press_button(executor, args, robot_idx)


def set_rotary_control(executor, args: dict, robot_idx: int) -> PredicateResult:
    value = _joint_value(executor, args["target_id"], args["control_id"])
    if value is None:
        return _fail("rotary control joint not found")
    return _success("rotary control state changed", joint_qpos=value, goal=args.get("goal"))


PREDICATES = {
    "navigate_to_fixture": navigate_to_fixture,
    "give_space": give_space,
    "pick_up_object": pick_up_object,
    "place_in_receptacle": place_in_receptacle,
    "place_on_object": place_on_object,
    "place_next_to": place_next_to,
    "place_on_surface": place_on_surface,
    "place_under": place_under,
    "open_hinged_part": open_hinged_part,
    "close_hinged_part": close_hinged_part,
    "open_sliding_part": open_sliding_part,
    "close_sliding_part": close_sliding_part,
    "press_button": press_button,
    "press_lever": press_lever,
    "set_rotary_control": set_rotary_control,
}


def evaluate_predicate(executor, step: dict, robot_idx: int) -> PredicateResult:
    tool = step.get("tool")
    predicate = PREDICATES.get(tool)
    if predicate is None:
        return _fail(f"no physical predicate registered for {tool!r}")
    try:
        return predicate(executor, step.get("args") or {}, robot_idx)
    except Exception as exc:
        return _fail(f"predicate error: {exc}")
