"""The generator and the evaluator must resolve the SAME tool set.

They used to diverge: build_task_definition_from_spec injected open_hinged_part
for fixtures with hinged parts, while task_registry read the raw declared spec.
That single divergence caused "Tool open_hinged_part is not allowed here",
KeyError: 'hinged', and the over-narrow give_space allowlist. The specs are now
materialised, so this test is what keeps them from drifting apart again.
"""
import unittest

from data_generation.task_level.tasks.specs import load_all_task_specs
from data_generation.task_level.tasks.specs.runtime import (
    build_task_definition_from_spec,
)
from training.bc_task_vlm.task_registry import (
    _camel_to_snake_case,
    get_task_metadata,
)


class SpecParity(unittest.TestCase):
    def test_generator_and_evaluator_agree(self):
        mismatches = []
        for spec in load_all_task_specs():
            validator = build_task_definition_from_spec(spec).validator_factory(None)
            generator = dict(getattr(validator, "allowed_tool_specs", {}) or {})
            evaluator = get_task_metadata(
                _camel_to_snake_case(spec.composite_task)
            ).allowed_tool_specs
            if set(generator) != set(evaluator):
                mismatches.append(
                    (spec.composite_task,
                     sorted(set(generator) - set(evaluator)),
                     sorted(set(evaluator) - set(generator)))
                )
        self.assertEqual(mismatches, [], f"tool sets drifted: {mismatches}")

    def test_give_space_covers_reachable_fixtures(self):
        """Any fixture you can navigate to must be one you can yield at.

        Otherwise two agents contending there have no legal way to resolve it;
        hot_dog_setup's unyieldable `counter` scored 0/120 for exactly this.
        """
        offenders = []
        for spec in load_all_task_specs():
            tools = get_task_metadata(
                _camel_to_snake_case(spec.composite_task)
            ).allowed_tool_specs
            nav = set((tools.get("navigate_to_fixture") or {}).get("allowed_fixture_ids") or [])
            give = set((tools.get("give_space") or {}).get("allowed_fixture_ids") or [])
            if nav and give and nav - give:
                offenders.append((spec.composite_task, sorted(nav - give)))
        self.assertEqual(offenders, [], f"unyieldable reachable fixtures: {offenders}")


if __name__ == "__main__":
    unittest.main()
