"""Closed-loop (live-sim) evaluation of tool-calling VLMs.

Implements training/bc_task_vlm/plans/live_sim_eval_plan.md: the model controls
both agents live in the MuJoCo sim under the v2 self-scheduled contract (the
model emits the acting agent and may call task_complete), success is judged by
the task env's own ``_check_success`` plus the symbolic FSM goal checker, and
visuals are re-rendered from current sim state every turn.

Backends:
  oracle      replay the expert trajectory's own steps (verification gate:
              expect ~100% success; no model involved)
  degenerate  emit garbage every turn (verification gate: expect 0% success,
              termination by rejections/budget, no crashes)
  hf          a local HF VLM (base weights or base+LoRA adapter)
  gemini      Vertex Gemini with native function calling

Example (oracle smoke):
  python -m training.bc_task_vlm.live_sim_eval --backend oracle \
    --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_tasks.json \
    --dataset-root training/bc_task_vlm/eval_data_subset \
    --output-dir training/bc_task_vlm/eval_runs/live_sim_oracle__heldout_tasks \
    --max-trajectories 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
from typing import Any

from training.bc_task_vlm.prompting import (
    build_messages,
    build_user_prompt,
)
from training.bc_task_vlm.schema_utils import (
    TASK_COMPLETE_TOOL_NAME,
    augment_tool_specs_for_agent_prediction,
    compact_json_dumps,
)
from training.bc_task_vlm.task_registry import AGENT_IDS, get_task_metadata
from training.bc_task_vlm.tool_calling import (
    build_tool_schemas,
    parse_first_qwen_tool_call,
    pop_agent_argument,
    tool_call_to_single_step_payload,
)

# Expert view-selection rule, measured over 162 trajectories (~3,300 calls):
# views deterministically match the NEXT action (scout set before navigate,
# wrist set before manipulation, overhead at the episode opening). The next
# action is unknown before generation, so the first pass renders the overhead
# set and the second pass re-renders the canonical set for the predicted
# action (see --two-pass-views).
OVERHEAD_VIEWS = ("top_view", "room_view", "map")
SCOUT_VIEWS = ("agentview_center", "agentview_left", "agentview_right")
WRIST_VIEWS = ("wrist", "agentview_center")
NAVIGATE_TOOLS = {"navigate_to_fixture"}
_NUMBERED_OBJECT_RE = re.compile(r"^obj_(\d+)$")


def canonical_views_for_tool(tool_name: str) -> tuple[str, ...]:
    if tool_name in NAVIGATE_TOOLS:
        return SCOUT_VIEWS
    if tool_name in {"communicate", TASK_COMPLETE_TOOL_NAME}:
        return OVERHEAD_VIEWS
    return WRIST_VIEWS


# ---------------------------------------------------------------------------
# FSM mirror: incremental legality + goal checking over symbolic state
# ---------------------------------------------------------------------------


class FsmMirror:
    """Replays the FiniteStateTaskValidator loop body one step at a time.

    Mirrors fsm.py::FiniteStateTaskValidator.validate's per-step behavior
    (communicate gate, allowed tools, preconditions, effects, goal check)
    against the TRAJECTORY's sampled initial state instead of the spec's
    template. Legality failures are returned, never raised.
    """

    def __init__(self, *, composite_task: str, trajectory: dict[str, Any]) -> None:
        from data_generation.task_level.tasks.specs import load_task_spec
        from data_generation.task_level.tasks.specs.runtime import (
            SpecDrivenTaskValidator,
        )

        spec = load_task_spec(composite_task)
        self.validator = SpecDrivenTaskValidator(spec)
        initial_state = trajectory.get("initial_state")
        if initial_state:
            self.validator.initial_state = deepcopy(initial_state)
        agents = self.validator._normalize_agents(trajectory.get("agents"))
        self.runtime_state = self.validator._build_runtime_state(agents)
        self.goal_satisfied = self.validator.is_goal_state_satisfied(
            self.runtime_state
        )

    def partial_goal_fraction(self) -> float:
        """Fraction of top-level spec goal conditions currently satisfied."""

        spec = self.validator._task_spec
        conditions = spec.goal_conditions
        if not conditions:
            return float(self.goal_satisfied)
        satisfied = 0
        for condition in conditions:
            original = self.validator._task_spec
            try:
                self.validator._task_spec = replace(
                    original, goal_conditions=(condition,)
                )
                satisfied += int(
                    self.validator.is_goal_state_satisfied(self.runtime_state)
                )
            finally:
                self.validator._task_spec = original
        return satisfied / len(conditions)

    def step(self, step: dict[str, Any]) -> tuple[bool, str | None]:
        """Applies one symbolic step; returns (legal, reason_if_not)."""

        validator = self.validator
        state = self.runtime_state
        try:
            if step["tool"] == "communicate":
                validator._validate_communicate_step(step)
                state.communicated_agents.add(step["agent"])
                validator.apply_task_effects(step, state)
            else:
                if state.communicated_agents != set(validator.agent_ids):
                    return False, (
                        "Both agents must communicate before the first task "
                        "action."
                    )
                if step["tool"] not in validator.allowed_tool_specs:
                    return False, f"Tool {step['tool']} is not allowed here."
                validator._validate_task_local_symbolic_constraints(step)
                validator._validate_generic_transition(step, state)
                validator.validate_task_preconditions(step, state)
                validator._apply_generic_effects(step, state)
                validator.apply_task_effects(step, state)
            self.goal_satisfied = validator.is_goal_state_satisfied(state)
            return True, None
        except Exception as exc:  # validator raises typed validation errors
            return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Sim session: one env per task, rewound per trajectory
# ---------------------------------------------------------------------------


class SimSession:
    def __init__(
        self,
        *,
        composite_task: str,
        sample_trajectory: dict[str, Any],
        layout: int,
        style: int,
        seed: int,
        gl_backend: str,
        render_size: int,
    ) -> None:
        from robocasa.utils.sim_tool_executor import SimToolExecutor
        from robocasa.utils.trajectory_pruning import (
            build_trajectory_pruning_config,
        )

        self.composite_task = composite_task
        pruning = build_trajectory_pruning_config(sample_trajectory, layout=layout)
        kwargs: dict[str, Any] = {}
        if any(pruning.get(key) for key in (
            "update_fxtr_cfg_dict",
            "trajectory_object_names",
            "trajectory_object_types",
            "trajectory_object_specs",
        )):
            kwargs.update(
                update_fxtr_cfg_dict=pruning.get("update_fxtr_cfg_dict"),
                trajectory_object_names=pruning.get("trajectory_object_names"),
                trajectory_object_types=pruning.get("trajectory_object_types"),
                trajectory_object_specs=pruning.get("trajectory_object_specs"),
            )
        self.executor = SimToolExecutor(
            task_name=composite_task,
            robots=2,
            layout=layout,
            style=style,
            seed=seed,
            gl_backend=gl_backend,
            render_width=render_size,
            render_height=render_size,
            robot_spawn="trajectory",
            **kwargs,
        )
        self._synchronize_pruned_task_cardinality(composite_task)

    def _synchronize_pruned_task_cardinality(self, composite_task: str) -> None:
        """Keep native task counters consistent with trajectory pruning.

        Some RoboCasa tasks sample a variable object count while fixtures are
        being initialized.  Trajectory pruning changes RNG consumption and can
        subsequently remove objects that are absent from the recorded initial
        state, leaving the task's cached count larger than ``env.objects``.
        BeverageOrganization's native checker indexes every object up to that
        cached count, so the mismatch otherwise produces ``KeyError: obj_N``.

        Keep this compatibility fix deliberately task-specific: similarly named
        counters in other tasks need not describe contiguous ``obj_N`` objects.
        """

        if composite_task != "BeverageOrganization":
            return
        indices = sorted(
            int(match.group(1))
            for name in self.executor.env.objects
            if (match := _NUMBERED_OBJECT_RE.fullmatch(str(name)))
        )
        if indices == list(range(len(indices))) and indices:
            self.executor.env.num_bev = len(indices)

    def start_trajectory(self, trajectory: dict[str, Any]):
        from robocasa.utils.trajectory_adapter import TrajectoryAdapter

        self.executor.restore_baseline_state()
        adapter = TrajectoryAdapter(
            executor=self.executor,
            allow_approximate_ids=True,
        )
        adapted = adapter.adapt(trajectory)
        self.executor.load_initial_state(adapted["initial_state"])
        # load_initial_state may prune a sampled variable-cardinality object
        # set, so counters derived during env construction must be repaired
        # after loading each trajectory rather than only once in __init__.
        self._synchronize_pruned_task_cardinality(self.composite_task)
        return adapter, adapted

    def native_success(self) -> tuple[bool | None, str | None]:
        """(success, error). Unwraps common env wrappers to find _check_success."""

        env = self.executor.env
        for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
            if candidate is None or not hasattr(candidate, "_check_success"):
                continue
            try:
                return bool(candidate._check_success()), None
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, f"no _check_success on {type(env).__name__}"

    def render_views(
        self,
        views: tuple[str, ...],
        *,
        agent_id: str,
        out_dir: Path,
        tag: str,
    ) -> tuple[list[str], list[str]]:
        """Renders the requested views to files; returns (paths, view_names)."""

        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            str(out_dir / f"{tag}_{view}.jpg") for view in views
        ]
        result = self.executor.get_image(
            views=list(views),
            image_paths=image_paths,
            agent_id=agent_id,
        )
        produced = result.details.get("image_paths") or image_paths
        return [str(p) for p in produced], list(views)

    def close(self) -> None:
        try:
            self.executor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class OraclePolicy:
    """Replays the expert trajectory's own action steps."""

    def __init__(self, raw_steps: list[dict[str, Any]]) -> None:
        self._steps = [s for s in raw_steps if s.get("tool") != "get_image"]
        self._cursor = 0

    def next_step(self, **_ignored) -> dict[str, Any] | None:
        if self._cursor >= len(self._steps):
            return None
        step = self._steps[self._cursor]
        self._cursor += 1
        return {
            "agent": step["agent"],
            "tool": step["tool"],
            "args": deepcopy(step.get("args", {})),
        }


class DegeneratePolicy:
    """Emits an unparseable/illegal call every turn."""

    def next_step(self, **_ignored) -> dict[str, Any] | None:
        return {
            "agent": "agent_0",
            "tool": "pick_up_object",
            "args": {"object_id": "nonexistent_object_xyz", "source_id": "counter"},
        }


class HfPolicy:
    """One local HF VLM shared across trajectories; generates one step/turn."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from training.bc_task_vlm.evaluation import VisionGenerationCollator

        self._torch = torch
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path, **model_kwargs
        )
        if args.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, str(args.adapter_path)
            )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.collator = VisionGenerationCollator(
            processor_name_or_path=args.model_name_or_path,
            max_length=args.max_length,
            trust_remote_code=False,
            sft_format="tool_call",
            image_resolution=args.image_resolution,
        )
        from transformers import AutoProcessor as _AP

        self.processor = _AP.from_pretrained(args.model_name_or_path)
        self.max_new_tokens = args.max_new_tokens

    def generate(self, feature: dict[str, Any]) -> str:
        torch = self._torch
        batch = self.collator([feature])
        batch.pop("sample_metadata", None)
        device = next(self.model.parameters()).device
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **batch, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[0][batch["input_ids"].shape[1] :]
        return self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class GeminiPolicy:
    """Vertex Gemini native-function-calling policy shared across turns."""

    def __init__(self, args: argparse.Namespace) -> None:
        import os

        from data_generation.task_level.runtime.client import (
            build_raw_google_genai_client,
            load_dotenv_file,
        )

        load_dotenv_file()
        location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        self.client = build_raw_google_genai_client(
            args.project, location, timeout_sec=args.request_timeout
        )
        self.model = args.model
        self.temperature = args.temperature
        self.max_output_tokens = args.max_output_tokens
        self.thinking_budget = args.thinking_budget
        self.max_retries = args.max_retries

    def generate(self, feature: dict[str, Any]) -> str:
        from google.genai import types

        from training.bc_task_vlm.eval_standalone import (
            _GEMINI_MIME_BY_SUFFIX,
            _message_text,
            build_vertex_function_declarations,
        )

        messages = feature["messages"]
        parts: list[Any] = []
        for image_path in feature["image_paths"]:
            path = Path(image_path)
            parts.append(types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=_GEMINI_MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg"),
            ))
        parts.append(types.Part.from_text(text=_message_text(messages[-1])))
        declarations = [
            types.FunctionDeclaration(**item)
            for item in build_vertex_function_declarations(
                feature["allowed_tool_specs"], include_agent_param=True
            )
        ]
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": _message_text(messages[0]),
            "tools": [types.Tool(function_declarations=declarations)],
            "tool_config": types.ToolConfig(function_calling_config=
                types.FunctionCallingConfig(mode="ANY")),
        }
        if self.thinking_budget >= 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                for candidate in getattr(response, "candidates", None) or []:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        call = getattr(part, "function_call", None)
                        if call is not None and getattr(call, "name", None):
                            return (
                                "<tool_call>\n"
                                + json.dumps({"name": call.name, "arguments": dict(call.args or {})})
                                + "\n</tool_call>"
                            )
                raise ValueError("Gemini response contained no function call")
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error


# ---------------------------------------------------------------------------
# The live loop for one trajectory
# ---------------------------------------------------------------------------


def run_trajectory(
    *,
    session: SimSession,
    policy,
    args: argparse.Namespace,
    task_name: str,
    composite_task: str,
    trajectory: dict[str, Any],
    frames_dir: Path | None,
) -> dict[str, Any]:
    task_metadata = get_task_metadata(task_name)
    train_get_image = bool(getattr(args, "train_get_image", False))
    tool_specs = augment_tool_specs_for_agent_prediction(
        task_metadata.allowed_tool_specs, include_get_image=train_get_image
    )
    tool_schemas = build_tool_schemas(
        agent_ids=AGENT_IDS,
        allowed_tool_specs=tool_specs,
        include_agent_param=True,
    )
    adapter, adapted = session.start_trajectory(trajectory)
    mirror = FsmMirror(composite_task=composite_task, trajectory=trajectory)
    if frames_dir is not None:
        session.executor.save_scene_frames(str(frames_dir), prefix="step_-001")

    expert_action_steps = [
        s for s in trajectory["steps"] if s.get("tool") != "get_image"
    ]
    budget_multiplier = 2 if train_get_image else 1
    step_budget = max(
        4, int(len(expert_action_steps) * args.step_budget_factor * budget_multiplier)
    )
    is_model_policy = isinstance(policy, (HfPolicy, GeminiPolicy))

    history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consecutive_rejections = 0
    declared_complete = False
    termination = "budget_exhausted"
    last_executed_tool: str | None = None
    step_index = 0
    requested_views: tuple[str, ...] | None = None
    requested_agent: str | None = None

    while step_index < step_budget:
        views_used: tuple[str, ...] | None = None
        proposal_started = time.perf_counter()
        if is_model_policy:
            proposal, views_used = _model_propose_step(
                session=session,
                policy=policy,
                args=args,
                task_metadata=task_metadata,
                tool_specs=tool_specs,
                tool_schemas=tool_schemas,
                trajectory=trajectory,
                history=history,
                step_index=step_index,
                last_executed_tool=last_executed_tool,
                frames_dir=frames_dir,
                requested_views=requested_views,
                requested_agent=requested_agent,
            )
            requested_views = None
            requested_agent = None
        else:
            proposal = policy.next_step()
        if proposal is None:
            termination = "policy_exhausted"
            break

        record: dict[str, Any] = {
            "step_index": step_index,
            "proposal": deepcopy(proposal),
            "views": list(views_used) if views_used else None,
            "proposal_elapsed_s": round(time.perf_counter() - proposal_started, 3),
        }
        if "error" in proposal:
            # Unparseable model output: no-op with feedback.
            record.update(legal=False, executed=False, reason=proposal["error"])
            history.append(
                {
                    "step": step_index,
                    "agent": proposal.get("agent") or "agent_0",
                    "tool": proposal.get("tool") or "invalid",
                    "args": proposal.get("args") or {},
                    "error": proposal["error"][:120],
                }
            )
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            step_index += 1
            continue

        if proposal["tool"] == TASK_COMPLETE_TOOL_NAME:
            declared_complete = True
            record.update(legal=True, executed=False, reason=None)
            records.append(record)
            termination = "task_complete_declared"
            break

        if proposal["tool"] == "get_image":
            views = tuple(proposal.get("args", {}).get("views", ()))
            known_views = set(OVERHEAD_VIEWS + SCOUT_VIEWS + WRIST_VIEWS)
            if not views or any(view not in known_views for view in views):
                record.update(
                    legal=False,
                    executed=False,
                    reason="invalid observation views",
                )
                consecutive_rejections += 1
            else:
                record.update(legal=True, executed=True, reason=None)
                history.append(
                    {
                        "step": step_index,
                        "agent": proposal["agent"],
                        "tool": "get_image",
                        "args": {"views": list(views)},
                    }
                )
                requested_views = views
                requested_agent = proposal["agent"]
                consecutive_rejections = 0
            records.append(record)
            step_index += 1
            continue

        symbolic_step = {
            "step": step_index,
            "agent": proposal["agent"],
            "tool": proposal["tool"],
            "args": deepcopy(proposal["args"]),
        }
        symbolic_state_before = deepcopy(mirror.runtime_state)
        symbolic_goal_before = mirror.goal_satisfied
        legal, reason = mirror.step(symbolic_step)
        record.update(legal=legal, reason=reason)

        # Only the oracle (expert replay) bypasses the legality gate; model
        # and degenerate policies are subject to it.
        should_execute = legal or isinstance(policy, OraclePolicy)
        if not should_execute:
            record["executed"] = False
            history.append({**symbolic_step, "error": (reason or "illegal")[:120]})
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            step_index += 1
            continue

        try:
            tool_call = adapter._adapt_step(
                symbolic_step,
                resolved_initial_state=adapted["initial_state"],
                output_dir=None,
            )
            result = session.executor.execute(
                tool_call["tool"],
                robot_idx=tool_call.get("robot_idx", 0),
                **tool_call.get("args", {}),
            )
            record.update(
                executed=True,
                sim_success=bool(result.success),
            )
            if not result.success:
                mirror.runtime_state = symbolic_state_before
                mirror.goal_satisfied = symbolic_goal_before
                details = getattr(result, "details", None) or {}
                sim_reason = str(
                    details.get("error")
                    or details.get("reason")
                    or "simulator reported an unsuccessful tool call"
                )
                record["sim_error"] = sim_reason
                history.append({**symbolic_step, "error": sim_reason[:120]})
                consecutive_rejections += 1
                records.append(record)
                if consecutive_rejections >= args.max_consecutive_rejections:
                    termination = "max_consecutive_rejections"
                    break
                step_index += 1
                continue
        except Exception as exc:
            mirror.runtime_state = symbolic_state_before
            mirror.goal_satisfied = symbolic_goal_before
            record.update(
                executed=False,
                sim_success=False,
                sim_error=f"{type(exc).__name__}: {exc}",
            )
            history.append(
                {**symbolic_step, "error": f"{type(exc).__name__}: {exc}"[:120]}
            )
            consecutive_rejections += 1
            records.append(record)
            if consecutive_rejections >= args.max_consecutive_rejections:
                termination = "max_consecutive_rejections"
                break
            step_index += 1
            continue

        consecutive_rejections = 0
        last_executed_tool = symbolic_step["tool"]
        history.append(symbolic_step)
        if frames_dir is not None:
            try:
                session.executor.save_scene_frames(
                    str(frames_dir), prefix=f"step_{step_index:03d}"
                )
            except Exception:
                pass

        record["fsm_goal"] = mirror.goal_satisfied
        native_now, native_err = session.native_success()
        record["native_success"] = native_now
        if native_err:
            record["native_error"] = native_err
        records.append(record)

        if mirror.goal_satisfied and native_now:
            termination = "goal_satisfied"
            step_index += 1
            break
        step_index += 1

    native, native_error = session.native_success()
    fsm_goal = mirror.goal_satisfied
    return {
        "task_name": task_name,
        "composite_task": composite_task,
        "trajectory_id": trajectory.get("trajectory_id"),
        "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
        "expert_steps": len(expert_action_steps),
        "steps_used": len(records),
        "executed_steps": sum(1 for r in records if r.get("executed")),
        "rejected_steps": sum(
            1
            for r in records
            if r.get("legal") is False or r.get("sim_success") is False
        ),
        "termination": termination,
        "declared_complete": declared_complete,
        "native_success": native,
        "native_error": native_error,
        "fsm_goal_satisfied": fsm_goal,
        "partial_goal_fraction": mirror.partial_goal_fraction(),
        "step_efficiency_ratio": (
            len(records) / len(expert_action_steps) if expert_action_steps else None
        ),
        "agent_turns": [
            r["proposal"].get("agent") for r in records if r.get("executed")
        ],
        "steps": records,
    }


def _model_propose_step(
    *,
    session: SimSession,
    policy: "HfPolicy | GeminiPolicy",
    args: argparse.Namespace,
    task_metadata,
    tool_specs: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    trajectory: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    last_executed_tool: str | None,
    frames_dir: Path | None,
    requested_views: tuple[str, ...] | None = None,
    requested_agent: str | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Two-phase turn: overhead render -> predict; re-render canonical views
    for the predicted tool and re-predict once if they differ."""

    render_dir = (frames_dir or Path(args.output_dir) / "_tmp_views")
    if getattr(args, "train_get_image", False):
        views = requested_views or ()
        proposal = _generate_once(
            session=session,
            policy=policy,
            args=args,
            task_metadata=task_metadata,
            tool_specs=tool_specs,
            tool_schemas=tool_schemas,
            trajectory=trajectory,
            history=history,
            step_index=step_index,
            views=views,
            render_dir=render_dir,
            agent_hint=requested_agent,
        )
        return proposal, views
    views = OVERHEAD_VIEWS
    proposal = _generate_once(
        session=session,
        policy=policy,
        args=args,
        task_metadata=task_metadata,
        tool_specs=tool_specs,
        tool_schemas=tool_schemas,
        trajectory=trajectory,
        history=history,
        step_index=step_index,
        views=views,
        render_dir=render_dir,
    )
    if not args.two_pass_views or "error" in proposal:
        return proposal, views
    canonical = canonical_views_for_tool(proposal["tool"])
    if canonical == views:
        return proposal, views
    second = _generate_once(
        session=session,
        policy=policy,
        args=args,
        task_metadata=task_metadata,
        tool_specs=tool_specs,
        tool_schemas=tool_schemas,
        trajectory=trajectory,
        history=history,
        step_index=step_index,
        views=canonical,
        render_dir=render_dir,
        agent_hint=proposal.get("agent"),
    )
    if "error" in second:
        return proposal, views
    return second, canonical


def _generate_once(
    *,
    session: SimSession,
    policy: "HfPolicy | GeminiPolicy",
    args: argparse.Namespace,
    task_metadata,
    tool_specs: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    trajectory: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    views: tuple[str, ...],
    render_dir: Path,
    agent_hint: str | None = None,
) -> dict[str, Any]:
    if views:
        image_paths, view_names = session.render_views(
            views,
            agent_id=agent_hint or "agent_0",
            out_dir=render_dir,
            tag=f"turn_{step_index:03d}_{len(views)}v",
        )
    else:
        image_paths, view_names = [], []
    user_prompt = build_user_prompt(
        composite_task=task_metadata.composite_task,
        task_instruction=trajectory.get("task", ""),
        agent_id="",
        next_step_index=step_index,
        observation_views=view_names,
        history_steps=history,
        allowed_tool_specs=tool_specs,
        sft_format="tool_call",
        predict_agent=True,
    )
    feature = {
        "sample_id": f"live/{trajectory.get('trajectory_id')}/turn_{step_index}",
        "messages": build_messages(
            user_prompt=user_prompt,
            num_images=len(image_paths),
            predict_agent=True,
            train_get_image=bool(getattr(args, "train_get_image", False)),
        ),
        "image_paths": image_paths,
        "tool_schemas": tool_schemas,
        "allowed_tool_specs": tool_specs,
    }
    from training.bc_task_vlm.prompting import append_few_shot_block

    if args.task_spec_detail:
        feature["messages"] = append_few_shot_block(
            feature["messages"], args.task_spec_blocks[task_metadata.dataset_name]
        )
    if args.few_shot:
        feature["messages"] = append_few_shot_block(
            feature["messages"], args.few_shot_blocks[task_metadata.dataset_name]
        )
    try:
        decoded = policy.generate(feature)
        parsed = parse_first_qwen_tool_call(decoded)
        agent, stripped = pop_agent_argument(parsed)
        if agent not in AGENT_IDS:
            return {"error": f'missing/invalid "agent" argument: {agent!r}'}
        payload = tool_call_to_single_step_payload(
            stripped,
            step_index=step_index,
            agent_id=agent,
            agent_ids=AGENT_IDS,
            allowed_tool_specs=tool_specs,
        )
        step = payload["steps"][0]
        return {"agent": step["agent"], "tool": step["tool"], "args": step["args"]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("oracle", "degenerate", "hf", "gemini"), required=True
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save per-camera stills for every evaluated trajectory.",
    )
    parser.add_argument("--record-fps", type=int, default=2)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gl-backend", default="osmesa")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--step-budget-factor", type=float, default=2.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=3)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-firsts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record stills for the first success and first failure per task.",
    )
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Local LoRA directory or Hugging Face adapter repository ID.",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sft-format", choices=("tool_call",), default="tool_call")
    parser.add_argument("--no-forced-json", action="store_true")
    parser.add_argument(
        "--train-get-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the model request observations with get_image.",
    )
    parser.add_argument("--task-spec-detail", action="store_true")
    parser.add_argument("--few-shot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model", default="gemini-3.5-flash-preview")
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument(
        "--two-pass-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-render canonical views for the predicted tool and re-predict.",
    )
    return parser.parse_args()


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    trajectory_ids_by_task: dict[str, list[str]] = {
        task: sorted(ids)
        for task, ids in manifest["trajectory_ids_by_task"].items()
    }
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",")}
        trajectory_ids_by_task = {
            t: ids for t, ids in trajectory_ids_by_task.items() if t in keep
        }
        if not trajectory_ids_by_task:
            available = ", ".join(sorted(manifest["trajectory_ids_by_task"]))
            raise SystemExit(
                f"--tasks matched no manifest tasks. Available tasks: {available}"
            )

    from training.bc_task_vlm.eval_standalone import (
        _load_few_shot_blocks,
        _load_task_spec_blocks,
    )

    task_names = sorted(trajectory_ids_by_task)
    args.task_spec_blocks = (
        _load_task_spec_blocks(task_names) if args.task_spec_detail else {}
    )
    args.few_shot_blocks = _load_few_shot_blocks(task_names) if args.few_shot else {}

    results_path = args.output_dir / "live_sim_trajectories.jsonl"
    done: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["task_name"], r["trajectory_id"]))

    policy = None
    policy_started = time.perf_counter()
    if args.backend == "hf":
        if not args.model_name_or_path:
            raise SystemExit("--backend hf requires --model-name-or-path")
        policy = HfPolicy(args)
    elif args.backend == "gemini":
        policy = GeminiPolicy(args)
    elif args.backend == "degenerate":
        policy = DegeneratePolicy()
    policy_load_s = round(time.perf_counter() - policy_started, 3)
    print(f"[timing] policy_load_s={policy_load_s}", flush=True)

    total_done = len(done)
    firsts: dict[tuple[str, bool], bool] = {}
    if args.record_firsts:
        for task_name in trajectory_ids_by_task:
            for success, label in ((True, "success"), (False, "failure")):
                firsts[(task_name, success)] = (
                    args.output_dir / "recordings" / task_name / label
                ).exists()
    with results_path.open("a", encoding="utf-8") as out:
        for task_name, trajectory_ids in trajectory_ids_by_task.items():
            task_metadata = get_task_metadata(task_name)
            composite = task_metadata.composite_task
            session = None
            for trajectory_id in trajectory_ids:
                if (task_name, trajectory_id) in done:
                    continue
                if (
                    args.max_trajectories is not None
                    and total_done >= args.max_trajectories
                ):
                    break
                traj_path = (
                    args.dataset_root
                    / task_metadata.dataset_name
                    / trajectory_id
                    / "original_trajectory.json"
                )
                trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
                if session is None:
                    print(f"[{task_name}] building sim session...", flush=True)
                    session_started = time.perf_counter()
                    try:
                        session = SimSession(
                            composite_task=composite,
                            sample_trajectory=trajectory,
                            layout=args.layout,
                            style=args.style,
                            seed=args.seed,
                            gl_backend=args.gl_backend,
                            render_size=args.render_size,
                        )
                        session_build_s = round(
                            time.perf_counter() - session_started, 3
                        )
                        print(
                            f"[{task_name}] session_build_s={session_build_s}",
                            flush=True,
                        )
                    except Exception as exc:
                        result = {
                            "task_name": task_name,
                            "composite_task": composite,
                            "trajectory_id": trajectory_id,
                            "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
                            "termination": "harness_error",
                            "native_success": None,
                            "fsm_goal_satisfied": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-2000:],
                        }
                        out.write(json.dumps(result, ensure_ascii=True) + "\n")
                        out.flush()
                        total_done += 1
                        print(
                            f"[{task_name}/{trajectory_id}] session build failed: "
                            f"{result['error']}", flush=True
                        )
                        break
                frames_dir = None
                if args.save_frames:
                    frames_dir = args.output_dir / "frames" / task_name / trajectory_id
                elif args.record_firsts:
                    # Record until both a success and a failure exist for the
                    # task; the final classification renames the directory.
                    if not (
                        firsts.get((task_name, True))
                        and firsts.get((task_name, False))
                    ):
                        frames_dir = (
                            args.output_dir
                            / "recordings"
                            / task_name
                            / trajectory_id
                        )
                started = time.time()
                try:
                    if args.backend == "oracle":
                        trajectory_policy = OraclePolicy(trajectory["steps"])
                    else:
                        trajectory_policy = policy
                    result = run_trajectory(
                        session=session,
                        policy=trajectory_policy,
                        args=args,
                        task_name=task_name,
                        composite_task=composite,
                        trajectory=trajectory,
                        frames_dir=frames_dir,
                    )
                except Exception as exc:
                    result = {
                        "task_name": task_name,
                        "composite_task": composite,
                        "trajectory_id": trajectory_id,
                        "termination": "harness_error",
                        "native_success": None,
                        "fsm_goal_satisfied": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                result["elapsed_s"] = round(time.time() - started, 1)
                if "session_build_s" in locals() and session_build_s is not None:
                    result["session_build_s"] = session_build_s
                    session_build_s = None
                out.write(json.dumps(result, ensure_ascii=True) + "\n")
                out.flush()
                total_done += 1
                success = bool(result.get("native_success"))
                if frames_dir is not None and not firsts.get((task_name, success)):
                    firsts[(task_name, success)] = True
                    if args.record_firsts and not args.save_frames:
                        classified = (
                            args.output_dir / "recordings" / task_name /
                            ("success" if success else "failure")
                        )
                        if classified.exists():
                            shutil.rmtree(classified)
                        frames_dir.rename(classified)
                        _assemble_recording_videos(classified, fps=args.record_fps)
                elif (
                    frames_dir is not None
                    and args.record_firsts
                    and not args.save_frames
                    and frames_dir.exists()
                ):
                    shutil.rmtree(frames_dir)
                native_error_text = (
                    f" native_error={result['native_error']}"
                    if result.get("native_error")
                    else ""
                )
                print(
                    f"[{task_name}/{trajectory_id}] "
                    f"native={result.get('native_success')}"
                    f"{native_error_text} "
                    f"fsm={result.get('fsm_goal_satisfied')} "
                    f"term={result.get('termination')} "
                    f"({result['elapsed_s']}s)",
                    flush=True,
                )
            if session is not None:
                session.close()

    _write_metrics(
        results_path,
        args.output_dir,
        policy_load_s=policy_load_s,
        total_elapsed_s=round(time.perf_counter() - run_started, 3),
    )


def _write_metrics(
    results_path: Path,
    output_dir: Path,
    *,
    policy_load_s: float | None = None,
    total_elapsed_s: float | None = None,
) -> None:
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(records)
    if not n:
        return
    native = sum(1 for r in records if r.get("native_success"))
    fsm = sum(1 for r in records if r.get("fsm_goal_satisfied"))
    agree = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and bool(r.get("native_success")) == bool(r.get("fsm_goal_satisfied"))
    )
    comparable = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and r.get("fsm_goal_satisfied") is not None
    )
    metrics = {
        "num_trajectories": n,
        "policy_load_s": policy_load_s,
        "total_elapsed_s": total_elapsed_s,
        "native_success_rate": native / n,
        "fsm_goal_rate": fsm / n,
        "judge_agreement_rate": agree / comparable if comparable else None,
        "declared_complete_rate": sum(
            1 for r in records if r.get("declared_complete")
        ) / n,
        "harness_error_rate": sum(
            1 for r in records if r.get("termination") == "harness_error"
        ) / n,
        "terminations": {},
        "mean_steps_used": sum(r.get("steps_used", 0) for r in records) / n,
        "mean_trajectory_elapsed_s": _mean_present(
            r.get("elapsed_s") for r in records
        ),
        "total_session_build_s": sum(
            r.get("session_build_s", 0.0) for r in records
        ),
        "mean_model_proposal_elapsed_s": _mean_present(
            step.get("proposal_elapsed_s")
            for r in records
            for step in r.get("steps", [])
        ),
        "mean_expert_steps": sum(r.get("expert_steps", 0) for r in records) / n,
        "mean_partial_goal_fraction": sum(
            r.get("partial_goal_fraction", 0.0) for r in records
        ) / n,
        "mean_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio") for r in records
        ),
        "mean_success_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio")
            for r in records
            if r.get("native_success")
        ),
        "rejection_rate": (
            sum(r.get("rejected_steps", 0) for r in records)
            / max(1, sum(r.get("steps_used", 0) for r in records))
        ),
        "agent_turn_counts": {},
        "same_agent_transition_rate": None,
        "same_agent_run_lengths": {},
    }
    turns = [turn for r in records for turn in r.get("agent_turns", [])]
    metrics["agent_turn_counts"] = {
        agent: turns.count(agent) for agent in AGENT_IDS
    }
    same = total_transitions = 0
    run_lengths: dict[str, int] = {}
    for record in records:
        trajectory_turns = record.get("agent_turns", [])
        total_transitions += max(0, len(trajectory_turns) - 1)
        same += sum(a == b for a, b in zip(trajectory_turns, trajectory_turns[1:]))
        for length in _run_lengths(trajectory_turns):
            key = str(length)
            run_lengths[key] = run_lengths.get(key, 0) + 1
    metrics["same_agent_transition_rate"] = (
        same / total_transitions if total_transitions else None
    )
    metrics["same_agent_run_lengths"] = run_lengths
    for r in records:
        term = r.get("termination", "unknown")
        metrics["terminations"][term] = metrics["terminations"].get(term, 0) + 1
    (output_dir / "live_sim_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _mean_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _run_lengths(turns: list[str]) -> list[int]:
    if not turns:
        return []
    lengths: list[int] = []
    current = turns[0]
    length = 1
    for turn in turns[1:]:
        if turn == current:
            length += 1
        else:
            lengths.append(length)
            current, length = turn, 1
    lengths.append(length)
    return lengths


def _assemble_recording_videos(recording_dir: Path, *, fps: int) -> None:
    """Create one MP4 per camera from the saved step stills."""

    import imageio.v2 as imageio

    for camera_dir in sorted(path for path in recording_dir.iterdir() if path.is_dir()):
        frames = sorted(camera_dir.glob("*.jpg"))
        if not frames:
            continue
        writer = imageio.get_writer(recording_dir / f"{camera_dir.name}.mp4", fps=fps)
        try:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
