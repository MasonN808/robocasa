import unittest
from unittest.mock import patch

from data_generation.task_level.pipeline.sim_normalization import (
    FixtureSimulationMetadata,
    _load_task_simulation_snapshot,
    _normalize_token_for_fixture,
    _resolve_symbolic_fixture_metadata,
)


class TestSimNormalization(unittest.TestCase):
    def test_snapshot_uses_configured_gl_backend(self):
        captured_kwargs = {}

        class FakeEnv:
            @staticmethod
            def get_ep_meta():
                return {}

        class FakeExecutor:
            env = FakeEnv()

            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            @staticmethod
            def get_scene_description():
                return {"fixtures": {}, "objects": {}}

            @staticmethod
            def close():
                return None

        _load_task_simulation_snapshot.cache_clear()
        with (
            patch.dict("os.environ", {"MUJOCO_GL": "egl"}),
            patch(
                "data_generation.task_level.pipeline.sim_normalization._ordered_scene_candidates",
                return_value=((11, 14, 42),),
            ),
            patch(
                "data_generation.task_level.pipeline.sim_normalization.SimToolExecutor",
                FakeExecutor,
            ),
        ):
            _load_task_simulation_snapshot("ExampleTask", 2)
        _load_task_simulation_snapshot.cache_clear()

        self.assertEqual(captured_kwargs["gl_backend"], "egl")

    def test_anchor_parent_wins_over_generic_counter_role(self):
        payload = {
            "initial_state": {
                "fixtures": {
                    "cab": {"fixture_type": "cabinet"},
                    "counter": {"fixture_type": "counter"},
                    "cabinet_parent_counter": {"fixture_type": "counter"},
                },
                "objects": {},
                "agents": {},
            },
            "grounding": {
                "symbols": {
                    "cabinet_parent_counter": {
                        "entity_type": "fixture",
                        "resolver": "nearest_placeable_surface_to_fixture",
                        "anchor_fixture_symbol": "cab",
                        "preferred_fixture_types": ["counter"],
                    }
                }
            },
        }
        snapshot = {
            "scene": {
                "objects": {},
                "object_placements": {},
                "fixture_refs": {
                    "cab": "cab_1",
                    "counter": "counter_2",
                },
                "fixtures": {
                    "cab_1": {
                        "fixture_type": "cabinet",
                        "parent_fixture": "counter_1",
                    },
                    "counter_1": {"fixture_type": "counter"},
                    "counter_2": {"fixture_type": "counter"},
                },
            },
            "fixture_details": {
                "cab_1": {
                    "fixture_type": "cabinet",
                    "parent_fixture": "counter_1",
                },
                "counter_1": {"fixture_type": "counter"},
                "counter_2": {"fixture_type": "counter"},
            },
        }

        resolved = _resolve_symbolic_fixture_metadata(
            task_name="ExampleTask",
            payload=payload,
            snapshot=snapshot,
            object_aliases={},
        )

        self.assertEqual(
            resolved["cabinet_parent_counter"].concrete_id,
            "counter_1",
        )

    def test_normalize_timer_knob_to_time_control(self):
        fixture_metadata = {
            "toaster_oven": FixtureSimulationMetadata(
                symbol_id="toaster_oven",
                concrete_id="toaster_oven_main_group",
                fixture_type="toaster_oven",
                part_ids=(),
                control_ids=("time", "temperature", "function", "doneness"),
                support_site_ids=(),
            )
        }
        errors: list[str] = []

        resolved = _normalize_token_for_fixture(
            "timer_knob",
            fixture_id="toaster_oven",
            fixture_metadata_by_symbol=fixture_metadata,
            kind="control",
            location_aliases=None,
            errors=errors,
            context="trajectory.steps[0]",
        )

        self.assertEqual(resolved, "time")
        self.assertEqual(errors, [])

    def test_normalize_timer_knob_to_timer_control(self):
        fixture_metadata = {
            "oven": FixtureSimulationMetadata(
                symbol_id="oven",
                concrete_id="oven_main_group",
                fixture_type="oven",
                part_ids=(),
                control_ids=("timer", "temperature"),
                support_site_ids=(),
            )
        }
        errors: list[str] = []

        resolved = _normalize_token_for_fixture(
            "timer_knob",
            fixture_id="oven",
            fixture_metadata_by_symbol=fixture_metadata,
            kind="control",
            location_aliases=None,
            errors=errors,
            context="trajectory.steps[0]",
        )

        self.assertEqual(resolved, "timer")
        self.assertEqual(errors, [])
