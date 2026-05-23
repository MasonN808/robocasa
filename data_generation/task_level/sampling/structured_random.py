"""Implement seed-driven structured random sampling over base trajectories."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from data_generation.task_level.sampling.base import BaseSamplingStrategy
from data_generation.utils import stable_json_sha256

if TYPE_CHECKING:
    from data_generation.task_level.generation.raw.config import RuntimeConfig
    from data_generation.task_level.tasks import TaskDefinition
    from data_generation.task_level.tasks.base import TaskInstance


STRUCTURED_RANDOM_RANKS = ("low", "medium", "high")
STRUCTURED_RANDOM_FIELDS = (
    "communication_message_length",
    "communication_message_complexity",
    "tool_call_diversity",
)


class StructuredRandomSamplingStrategy(BaseSamplingStrategy):
    """Uses base sampling with a code-selected random seed and configuration."""

    name = "structured_random"

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds a base prompt conditioned on a structured random configuration."""

        base_prompt = super().build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            task_instance=task_instance,
            variation_key=variation_key,
            retry_feedback=retry_feedback,
        )
        _, configuration = _structured_random_seed_and_configuration(
            task_definition=task_definition,
            variation_key=variation_key,
        )
        configuration_text = ", ".join(
            f"{field_name}={configuration[field_name]}"
            for field_name in STRUCTURED_RANDOM_FIELDS
        )
        return (
            f"Structured random configuration: {configuration_text}\n\n"
            "Use the fixed configuration above for this single trajectory.\n"
            "- Return exactly one trajectory using the normal single-trajectory "
            "JSON format requested below.\n"
            "- Do not return a samples array, a responses array, or multiple "
            "trajectory candidates.\n"
            "- communication_message_length controls whether communicate "
            "messages are terse, sentence-length, or detailed.\n"
            "- communication_message_complexity controls whether communicate "
            "messages are simple status/request messages, include task "
            "constraints, or include multi-part rationale and contingency "
            "planning.\n"
            "- tool_call_diversity controls whether the trajectory uses a "
            "narrow, moderate, or broad task-relevant mix of allowed "
            "non-communication tools.\n"
            "- Keep all tool calls task-relevant; do not add unnecessary "
            "actions just to increase diversity.\n\n"
            f"{base_prompt}"
        )


def _structured_random_seed_and_configuration(
    *,
    task_definition: TaskDefinition,
    variation_key: str,
) -> tuple[int, dict[str, str]]:
    """Selects one deterministic random seed and configuration for a run."""

    run_key = _run_level_variation_key(variation_key)
    seed_hex = stable_json_sha256(
        {
            "sampling": "structured_random",
            "task": task_definition.composite_task,
            "run": run_key,
        }
    )[:16]
    seed = int(seed_hex, 16)
    rng = random.Random(seed)
    configuration = {
        field_name: rng.choice(STRUCTURED_RANDOM_RANKS)
        for field_name in STRUCTURED_RANDOM_FIELDS
    }
    return seed, configuration


def _run_level_variation_key(variation_key: str) -> str:
    """Strips retry-attempt suffixes so retries keep the same random config."""

    match = re.match(r"^(traj-\d+)-attempt-\d+$", variation_key)
    if match is None:
        return variation_key
    return match.group(1)
