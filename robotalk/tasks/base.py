"""Backward-compatible facade for shared task-level prompting and validation code."""

from __future__ import annotations

from robotalk.tasks.shared.constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    PLACE_LOCATION_ARG_NAMES,
    RELEASE_TOOL_NAMES,
)
from robotalk.tasks.shared.errors import (
    CommunicationStepSemanticValidationError,
    DuplicateTrajectoryValidationError,
    HeldObjectSemanticValidationError,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    InsufficientValidUniqueTrajectoriesValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    NavigationSemanticValidationError,
    ObjectStateSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    PlacementDestinationSemanticValidationError,
    PostGoalActionSemanticValidationError,
    ResponseFormatValidationError,
    TaskPreconditionSemanticValidationError,
    TaskSemanticValidationError,
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
)
from robotalk.tasks.shared.fsm import FiniteStateTaskValidator
from robotalk.tasks.shared.instances import (
    build_canonical_agents,
    build_randomized_fixture_task_instance,
    resolve_initial_position_fixture_ids,
)
from robotalk.tasks.shared.prompting import make_task_prompt_builder
from robotalk.tasks.shared.schema import build_task_response_schema
from robotalk.tasks.shared.state import (
    AgentRuntimeState,
    TaskRuntimeState,
)
from robotalk.tasks.shared.types import (
    PreflightTokenEstimate,
    TaskDefinition,
    TaskInstance,
    TaskPromptBuilder,
    TaskValidator,
)
