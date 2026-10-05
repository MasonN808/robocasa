"""Multi-robot fix for robosuite's mobile-base mounting.

``RobotModel.add_mobile_base`` creates an intermediate body named
``manipulator_mount`` without the robot's naming prefix, so a scene with two
mobile manipulators contains two bodies with the same name and MuJoCo rejects
it. This wrapper gives the body its prefixed name (``robot0_manipulator_mount``),
which is exactly what the patched robosuite fork used during the RoboTalk
experiments did; the resulting MJCF is identical.
"""

from __future__ import annotations

from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import find_elements

_UNPREFIXED = "manipulator_mount"


def _prefix_manipulator_mount(add_mobile_base):
    def add_mobile_base_with_prefixed_mount(self, mobile_base):
        add_mobile_base(self, mobile_base)
        mount = find_elements(
            root=self.worldbody,
            tags="body",
            attribs={"name": _UNPREFIXED},
            return_first=True,
        )
        if mount is not None:
            mount.set("name", self.correct_naming(_UNPREFIXED))

    add_mobile_base_with_prefixed_mount.__wrapped__ = add_mobile_base
    add_mobile_base_with_prefixed_mount._robocasa_prefixed_mount = True
    return add_mobile_base_with_prefixed_mount


def apply() -> None:
    """Install the wrapper once (a no-op on a robosuite that already prefixes)."""

    if not getattr(RobotModel.add_mobile_base, "_robocasa_prefixed_mount", False):
        RobotModel.add_mobile_base = _prefix_manipulator_mount(RobotModel.add_mobile_base)


apply()
