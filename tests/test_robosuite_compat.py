"""robocasa.utils.robosuite_compat must reproduce the patched robosuite fork."""

import inspect
import re
import textwrap

import pytest

import robocasa  # noqa: F401  (installs the wrapper)
from robosuite.models.bases.omron_mobile_base import OmronMobileBase
from robosuite.models.robots import compositional
from robosuite.models.robots import robot_model as robot_model_module
from robosuite.models.robots.robot_model import RobotModel

MOUNT = re.compile(
    r'new_element\(\s*"body",\s*(?:self\.correct_naming\(\s*"manipulator_mount"\s*\)|"manipulator_mount")\s*\)'
)
# The installed method, captured before any test monkeypatches it.
ORIGINAL = inspect.unwrap(RobotModel.add_mobile_base)


def _variant(prefixed: bool):
    """Rebuild add_mobile_base as upstream (unprefixed) or the fork (prefixed)."""

    source = textwrap.dedent(inspect.getsource(ORIGINAL))
    name = 'self.correct_naming("manipulator_mount")' if prefixed else '"manipulator_mount"'
    source, count = MOUNT.subn(f'new_element("body", {name})', source)
    assert count == 1, "robosuite's add_mobile_base changed; revisit robosuite_compat"
    namespace = dict(vars(robot_model_module))
    exec(compile(source, robot_model_module.__file__, "exec"), namespace)
    return namespace["add_mobile_base"]


def _robot_xml(monkeypatch, method, idn):
    monkeypatch.setattr(RobotModel, "add_mobile_base", method)
    robot = compositional.PandaOmron(idn=idn)
    robot.add_base(OmronMobileBase(idn=idn))
    return robot.get_xml()


@pytest.mark.parametrize("idn", [0, 1])
def test_wrapped_upstream_matches_the_fork(monkeypatch, idn):
    from robocasa.utils.robosuite_compat import _prefix_manipulator_mount

    fork = _robot_xml(monkeypatch, _variant(prefixed=True), idn)
    patched = _robot_xml(monkeypatch, _prefix_manipulator_mount(_variant(prefixed=False)), idn)
    assert patched == fork
    assert f'name="robot{idn}_manipulator_mount"' in fork
