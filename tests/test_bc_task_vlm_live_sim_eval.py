from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from training.bc_task_vlm import live_sim_eval
from training.bc_task_vlm.divergence_analysis import analyze
from training.bc_task_vlm.live_sim_eval import (
    OVERHEAD_VIEWS,
    ROUTER_VIEWS,
    SCOUT_VIEWS,
    WRIST_VIEWS,
    _canonical_symbolic_initial_state,
    _completed_trajectory_keys,
    _finalize_first_recording,
    _global_step_index_for_turn,
    _routing_views_for_step,
    _step_index_for_turn,
    _run_lengths,
    _write_metrics,
    canonical_views_for_tool,
)


def test_canonical_views_follow_tool_kind():
    assert canonical_views_for_tool("navigate_to_fixture") == SCOUT_VIEWS
    assert canonical_views_for_tool("communicate") == OVERHEAD_VIEWS
    assert canonical_views_for_tool("pick_up_object") == WRIST_VIEWS


def test_router_views_are_target_independent_mixed_bundle():
    assert ROUTER_VIEWS == ("top_view", "wrist", "agentview_center")
    assert ROUTER_VIEWS not in {OVERHEAD_VIEWS, SCOUT_VIEWS, WRIST_VIEWS}
    assert _routing_views_for_step(2) == OVERHEAD_VIEWS
    assert _routing_views_for_step(3) == OVERHEAD_VIEWS
    assert _routing_views_for_step(5) == ROUTER_VIEWS


def test_live_turns_use_training_compatible_global_step_indices():
    assert [_global_step_index_for_turn(i) for i in range(6)] == [2, 3, 5, 7, 9, 11]
    assert [
        _step_index_for_turn(i, active_observation=True) for i in range(6)
    ] == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="non-negative"):
        _global_step_index_for_turn(-1)


def test_communication_router_prediction_still_gets_second_pass(monkeypatch, tmp_path):
    calls = []
    proposals = iter(
        [
            {"agent": "agent_0", "tool": "communicate", "args": {}},
            {"agent": "agent_0", "tool": "communicate", "args": {}},
        ]
    )

    def fake_generate_once(**kwargs):
        calls.append(kwargs)
        return next(proposals)

    monkeypatch.setattr(live_sim_eval, "_generate_once", fake_generate_once)
    proposal, views = live_sim_eval._model_propose_step(
        session=object(),
        policy=object(),
        args=SimpleNamespace(output_dir=tmp_path, two_pass_views=True),
        task_metadata=object(),
        tool_specs={},
        tool_schemas=[],
        trajectory={},
        history=[],
        step_index=5,
        last_executed_tool=None,
        frames_dir=None,
    )

    assert proposal["tool"] == "communicate"
    assert [call["views"] for call in calls] == [ROUTER_VIEWS, OVERHEAD_VIEWS]
    assert calls[1]["agent_hint"] == "agent_0"
    assert views == OVERHEAD_VIEWS


def test_active_observation_uses_requested_views_once(monkeypatch, tmp_path):
    calls = []

    def fake_generate_once(**kwargs):
        calls.append(kwargs)
        return {"agent": "agent_1", "tool": "pick_up_object", "args": {}}

    monkeypatch.setattr(live_sim_eval, "_generate_once", fake_generate_once)
    proposal, views = live_sim_eval._model_propose_step(
        session=object(),
        policy=object(),
        args=SimpleNamespace(
            output_dir=tmp_path, train_get_image=True, two_pass_views=True
        ),
        task_metadata=object(),
        tool_specs={},
        tool_schemas=[],
        trajectory={},
        history=[],
        step_index=1,
        last_executed_tool=None,
        frames_dir=None,
        requested_views=("wrist", "agentview_center"),
        requested_agent="agent_1",
    )

    assert proposal["tool"] == "pick_up_object"
    assert views == ("wrist", "agentview_center")
    assert len(calls) == 1
    assert calls[0]["views"] == views
    assert calls[0]["agent_hint"] == "agent_1"


def test_causal_cache_without_active_agent_is_image_free(monkeypatch, tmp_path):
    calls = []

    def fake_generate_once(**kwargs):
        calls.append(kwargs)
        return {
            "agent": "agent_0",
            "tool": "get_image",
            "args": {"views": list(OVERHEAD_VIEWS)},
        }

    monkeypatch.setattr(live_sim_eval, "_generate_once", fake_generate_once)
    proposal, views = live_sim_eval._model_propose_step(
        session=object(),
        policy=object(),
        args=SimpleNamespace(
            output_dir=tmp_path,
            train_get_image=True,
            get_image_observation_mode="causal_cache",
        ),
        task_metadata=object(),
        tool_specs={},
        tool_schemas=[],
        trajectory={},
        history=[],
        step_index=1,
        last_executed_tool=None,
        frames_dir=None,
        cached_observations_by_agent={
            "agent_0": (["agent0.jpg"], ["top_view"]),
            "agent_1": (["agent1.jpg"], ["wrist"]),
        },
        active_observation_agent=None,
    )

    assert proposal["tool"] == "get_image"
    assert views == ()
    assert len(calls) == 1
    assert calls[0]["views"] == ()
    assert calls[0]["agent_hint"] is None
    assert calls[0]["pre_rendered_image_paths"] is None


def test_causal_cache_uses_only_active_agent_cache_independent_of_step_index(
    monkeypatch, tmp_path
):
    calls = []

    def fake_generate_once(**kwargs):
        calls.append(kwargs)
        return {"agent": "agent_1", "tool": "communicate", "args": {}}

    agent_0_paths = ["agent0_top.jpg", "agent0_room.jpg", "agent0_map.jpg"]
    agent_1_paths = ["agent1_wrist.jpg", "agent1_center.jpg"]
    monkeypatch.setattr(live_sim_eval, "_generate_once", fake_generate_once)
    for step_index in (2, 3, 4, 5):
        proposal, views = live_sim_eval._model_propose_step(
            session=object(),
            policy=object(),
            args=SimpleNamespace(
                output_dir=tmp_path,
                train_get_image=True,
                get_image_observation_mode="causal_cache",
            ),
            task_metadata=object(),
            tool_specs={},
            tool_schemas=[],
            trajectory={},
            history=[],
            step_index=step_index,
            last_executed_tool=None,
            frames_dir=None,
            cached_observations_by_agent={
                "agent_0": (agent_0_paths, list(OVERHEAD_VIEWS)),
                "agent_1": (agent_1_paths, list(WRIST_VIEWS)),
            },
            active_observation_agent="agent_1",
        )

        assert proposal["tool"] == "communicate"
        assert views == WRIST_VIEWS

    assert len(calls) == 4
    assert all(call["views"] == WRIST_VIEWS for call in calls)
    assert all(call["agent_hint"] == "agent_1" for call in calls)
    assert all(
        call["pre_rendered_image_paths"] == agent_1_paths for call in calls
    )
    assert all(
        call["pre_rendered_image_paths"] != agent_0_paths for call in calls
    )


def test_causal_cache_records_only_requesting_agent():
    cached = {
        "agent_0": (["agent0_top.jpg"], ["top_view"]),
    }

    live_sim_eval._record_agent_observation(
        cached,
        agent_id="agent_1",
        image_paths=["agent1_top.jpg"],
        view_names=["top_view"],
    )

    assert cached == {
        "agent_0": (["agent0_top.jpg"], ["top_view"]),
        "agent_1": (["agent1_top.jpg"], ["top_view"]),
    }


def test_causal_cache_owner_gate_and_physical_invalidation():
    missing = live_sim_eval._causal_cache_rejection_reason(
        tool_name="navigate_to_fixture",
        proposed_agent="agent_0",
        active_observation_agent=None,
    )
    foreign = live_sim_eval._causal_cache_rejection_reason(
        tool_name="communicate",
        proposed_agent="agent_1",
        active_observation_agent="agent_0",
    )

    assert "requires a preceding" in missing
    assert "belongs to agent_0" in foreign
    assert (
        live_sim_eval._causal_cache_rejection_reason(
            tool_name="communicate",
            proposed_agent="agent_0",
            active_observation_agent="agent_0",
        )
        is None
    )
    assert (
        live_sim_eval._causal_cache_rejection_reason(
            tool_name="get_image",
            proposed_agent="agent_1",
            active_observation_agent="agent_0",
        )
        is None
    )
    assert live_sim_eval._tool_invalidates_active_observation(
        "navigate_to_fixture"
    )
    assert live_sim_eval._tool_invalidates_active_observation("pick_up_object")
    assert not live_sim_eval._tool_invalidates_active_observation("communicate")


def test_centralized_causal_cache_allows_only_cross_owner_communication():
    assert (
        live_sim_eval._causal_cache_rejection_reason(
            tool_name="communicate",
            proposed_agent="agent_0",
            active_observation_agent="agent_1",
            allow_cross_owner_communication=True,
        )
        is None
    )
    physical = live_sim_eval._causal_cache_rejection_reason(
        tool_name="pick_up_object",
        proposed_agent="agent_0",
        active_observation_agent="agent_1",
        allow_cross_owner_communication=True,
    )
    assert "belongs to agent_1" in physical


def test_both_causal_modes_use_prefix_selected_cache():
    assert live_sim_eval._uses_causal_cached_observations("causal_cache")
    assert live_sim_eval._uses_causal_cached_observations(
        "causal_cache_centralized"
    )
    assert not live_sim_eval._uses_causal_cached_observations("next_turn")


def test_pruned_organize_condiments_native_checker_omits_only_distractor():
    env = SimpleNamespace(
        objects={
            "condiment1": object(),
            "condiment2": object(),
            "condiment3": object(),
        },
        cab=object(),
    )
    inside = {"condiment1": True, "condiment2": True, "condiment3": True}
    far = {"condiment1": True, "condiment2": True, "condiment3": True}
    object_utils = SimpleNamespace(
        obj_inside_of=lambda _env, name, _cab: inside[name],
        gripper_obj_far=lambda _env, name: far[name],
    )

    assert live_sim_eval._check_pruned_organize_condiments_success(
        env, object_utils=object_utils
    )
    inside["condiment2"] = False
    assert not live_sim_eval._check_pruned_organize_condiments_success(
        env, object_utils=object_utils
    )
    del env.objects["condiment3"]
    with pytest.raises(KeyError, match="goal-relevant.*condiment3"):
        live_sim_eval._check_pruned_organize_condiments_success(
            env, object_utils=object_utils
        )


def test_native_success_routes_only_pruned_organize_condiments(monkeypatch):
    required = {
        "condiment1": object(),
        "condiment2": object(),
        "condiment3": object(),
    }
    pruned_env = SimpleNamespace(
        objects=required,
        _check_success=lambda: (_ for _ in ()).throw(
            AssertionError("upstream checker must not run for pruned scene")
        ),
    )
    session = object.__new__(live_sim_eval.SimSession)
    session.composite_task = "OrganizeCondiments"
    session.executor = SimpleNamespace(env=pruned_env)
    monkeypatch.setattr(
        live_sim_eval,
        "_check_pruned_organize_condiments_success",
        lambda _env: True,
    )
    assert session.native_success() == (True, None)

    unpruned_env = SimpleNamespace(
        objects={**required, "distractor": object()},
        _check_success=lambda: False,
    )
    session.executor = SimpleNamespace(env=unpruned_env)
    assert session.native_success() == (False, None)


def test_centralized_causal_cache_run_state_transitions(monkeypatch, tmp_path):
    proposals = iter(
        [
            {
                "agent": "agent_0",
                "tool": "get_image",
                "args": {"views": ["top_view"]},
            },
            {
                "agent": "agent_1",
                "tool": "get_image",
                "args": {"views": ["top_view"]},
            },
            {"agent": "agent_0", "tool": "communicate", "args": {}},
            {"agent": "agent_1", "tool": "navigate_to_fixture", "args": {}},
            {"agent": "agent_0", "tool": "task_complete", "args": {}},
        ]
    )
    proposal_states = []
    render_calls = []

    class FakePolicy:
        def generate(self, _feature):
            raise AssertionError("the proposal helper is monkeypatched")

    class FakeMirror:
        def __init__(self, **_kwargs):
            self.runtime_state = {}
            self.goal_satisfied = False

        def step(self, _step):
            return True, None

        def partial_goal_fraction(self):
            return 0.0

    class FakeAdapter:
        def _adapt_step(self, symbolic_step, **_kwargs):
            return {
                "tool": symbolic_step["tool"],
                "robot_idx": 0,
                "args": {},
            }

    class FakeExecutor:
        def execute(self, _tool, **_kwargs):
            return SimpleNamespace(success=True, details={})

    class FakeSession:
        executor = FakeExecutor()

        def start_trajectory(self, _trajectory):
            return FakeAdapter(), {"initial_state": {}}

        def render_views(self, views, *, agent_id, **_kwargs):
            render_calls.append((tuple(views), agent_id))
            return [f"{agent_id}_top.jpg"], list(views)

        def native_success(self):
            return False, None

    def fake_model_propose_step(**kwargs):
        proposal_states.append(
            {
                "active": kwargs["active_observation_agent"],
                "cache_agents": sorted(kwargs["cached_observations_by_agent"]),
            }
        )
        active = kwargs["active_observation_agent"]
        cached_paths, views = live_sim_eval._cached_observation_for_agent(
            kwargs["cached_observations_by_agent"], active
        )
        assert (cached_paths is None) == (active is None)
        return next(proposals), views

    monkeypatch.setattr(
        live_sim_eval,
        "get_task_metadata",
        lambda _task: SimpleNamespace(allowed_tool_specs={}),
    )
    monkeypatch.setattr(
        live_sim_eval,
        "augment_tool_specs_for_agent_prediction",
        lambda specs, **_kwargs: specs,
    )
    monkeypatch.setattr(live_sim_eval, "build_tool_schemas", lambda **_kwargs: [])
    monkeypatch.setattr(live_sim_eval, "FsmMirror", FakeMirror)
    monkeypatch.setattr(live_sim_eval, "_model_propose_step", fake_model_propose_step)

    result = live_sim_eval.run_trajectory(
        session=FakeSession(),
        policy=FakePolicy(),
        args=SimpleNamespace(
            train_get_image=True,
            get_image_observation_mode="causal_cache_centralized",
            step_budget_factor=3.0,
            max_consecutive_rejections=3,
            output_dir=tmp_path,
            layout=0,
            style=0,
            seed=0,
        ),
        task_name="beverage_organization",
        composite_task="BeverageOrganization",
        trajectory={
            "trajectory_id": "traj_smoke",
            "task": "organize beverages",
            "steps": [
                {"tool": "get_image"},
                {"tool": "get_image"},
                {"tool": "communicate"},
                {"tool": "navigate_to_fixture"},
            ],
        },
        frames_dir=None,
    )

    assert [state["active"] for state in proposal_states] == [
        None,
        "agent_0",
        "agent_1",
        "agent_1",
        None,
    ]
    assert [state["cache_agents"] for state in proposal_states] == [
        [],
        ["agent_0"],
        ["agent_0", "agent_1"],
        ["agent_0", "agent_1"],
        ["agent_0", "agent_1"],
    ]
    assert render_calls == [
        (("top_view",), "agent_0"),
        (("top_view",), "agent_1"),
    ]
    cross_owner_communication = result["steps"][2]
    assert cross_owner_communication["legal"] is True
    assert cross_owner_communication["executed"] is True
    assert result["termination"] == "task_complete_declared"


def test_generate_once_reuses_cached_files_without_rendering(monkeypatch, tmp_path):
    captured = {}

    class FakeSession:
        def render_views(self, *_args, **_kwargs):
            raise AssertionError("cached observations must not be re-rendered")

    class FakePolicy:
        def generate(self, feature):
            captured.update(feature)
            raise RuntimeError("stop after feature capture")

    monkeypatch.setattr(live_sim_eval, "build_user_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(live_sim_eval, "build_messages", lambda **_kwargs: [])

    cached_paths = ["cached_top.jpg", "cached_room.jpg", "cached_map.jpg"]
    proposal = live_sim_eval._generate_once(
        session=FakeSession(),
        policy=FakePolicy(),
        args=SimpleNamespace(
            output_dir=tmp_path,
            task_spec_detail=False,
            few_shot=False,
            train_get_image=True,
        ),
        task_metadata=SimpleNamespace(
            composite_task="BeverageOrganization",
            dataset_name="beverage_organization",
        ),
        tool_specs={},
        tool_schemas=[],
        trajectory={"trajectory_id": "traj_000002", "task": "organize"},
        history=[],
        step_index=2,
        views=OVERHEAD_VIEWS,
        render_dir=tmp_path,
        agent_hint="agent_0",
        pre_rendered_image_paths=cached_paths,
    )

    assert proposal == {"error": "RuntimeError: stop after feature capture"}
    assert captured["image_paths"] == cached_paths
    assert captured["agent_id"] == "agent_0"


def test_run_lengths():
    assert _run_lengths([]) == []
    assert _run_lengths(["agent_0", "agent_0", "agent_1", "agent_0"]) == [2, 1, 1]


def test_live_generation_feature_includes_collator_metadata(monkeypatch, tmp_path):
    captured = {}
    message_kwargs = {}

    class FakeSession:
        def render_views(self, *_args, **_kwargs):
            return [], []

    class FakePolicy:
        def generate(self, feature):
            captured.update(feature)
            raise RuntimeError("stop after feature capture")

    monkeypatch.setattr(live_sim_eval, "build_user_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(
        live_sim_eval,
        "build_messages",
        lambda **kwargs: message_kwargs.update(kwargs) or [],
    )

    proposal = live_sim_eval._generate_once(
        session=FakeSession(),
        policy=FakePolicy(),
        args=SimpleNamespace(
            output_dir=tmp_path,
            task_spec_detail=False,
            few_shot=False,
        ),
        task_metadata=SimpleNamespace(
            composite_task="BeverageOrganization",
            dataset_name="beverage_organization",
        ),
        tool_specs={},
        tool_schemas=[],
        trajectory={"trajectory_id": "traj_000002", "task": "organize"},
        history=[],
        step_index=3,
        views=OVERHEAD_VIEWS,
        render_dir=tmp_path,
    )

    assert proposal == {"error": "RuntimeError: stop after feature capture"}
    assert captured["task_name"] == "beverage_organization"
    assert captured["trajectory_id"] == "traj_000002"
    assert captured["step_index"] == 3
    assert captured["agent_id"] == ""
    assert captured["target_payload"] is None
    assert captured["target_tool_call"] is None
    assert captured["target_text"] == ""
    assert message_kwargs["target_text"] == ""


def test_symbolic_initial_state_normalizes_unique_legacy_types():
    task_spec = SimpleNamespace(
        initial_state={
            "agents": {"agent_0": {"location": "counter"}},
            "objects": {
                "hotdog_bun": {"object_type": "hotdog_bun", "location": "counter"},
                "plate": {"object_type": "plate", "location": "dining_table"},
            },
            "fixtures": {
                "counter": {"fixture_type": "counter"},
                "dining_table": {"fixture_type": "dining_counter"},
            },
        },
        grounding={
            "symbols": {
                "hotdog_bun": {"entity_type": "object", "object_type": "hotdog_bun"},
                "plate": {"entity_type": "object", "object_type": "plate"},
                "counter": {"entity_type": "fixture", "fixture_type": "counter"},
                "dining_table": {"entity_type": "fixture", "fixture_type": "dining_counter"},
            }
        },
    )
    trajectory = {
        "initial_state": {
            "agents": {"agent_0": {"location": "bun_source_fixture"}},
            "objects": {
                "bun": {"object_type": "hotdog_bun", "location": "bun_source_fixture"},
                "serving_plate": {"object_type": "plate", "location": "serving_surface"},
            },
            "fixtures": {
                "bun_source_fixture": {"fixture_type": "counter"},
                "serving_surface": {"fixture_type": "dining_table"},
            },
        },
        "grounding_map": {
            "symbols": {
                "serving_surface": {
                    "entity_type": "fixture",
                    "fixture_type": "dining_table",
                    "preferred_fixture_types": ["dining_counter"],
                }
            }
        },
    }
    normalized = _canonical_symbolic_initial_state(
        task_spec=task_spec, trajectory=trajectory
    )
    assert set(normalized["objects"]) == {"hotdog_bun", "plate"}
    assert set(normalized["fixtures"]) == {"counter", "dining_table"}
    assert normalized["objects"]["hotdog_bun"]["location"] == "counter"
    assert normalized["agents"]["agent_0"]["location"] == "counter"


def test_run_trajectory_initializes_fsm_from_symbolic_state(monkeypatch):
    original = {"initial_state": {"objects": {"bun": {}}}}
    adapted = {"initial_state": {"objects": {"hotdog_bun": {}}}}
    captured = {}

    class FakeSession:
        def start_trajectory(self, trajectory):
            assert trajectory is original
            return object(), adapted

    def capture_mirror(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after mirror capture")

    monkeypatch.setattr(
        live_sim_eval,
        "get_task_metadata",
        lambda _task: SimpleNamespace(allowed_tool_specs={}),
    )
    monkeypatch.setattr(
        live_sim_eval, "augment_tool_specs_for_agent_prediction", lambda specs, **_kwargs: specs
    )
    monkeypatch.setattr(live_sim_eval, "build_tool_schemas", lambda **_kwargs: [])
    monkeypatch.setattr(live_sim_eval, "FsmMirror", capture_mirror)

    with pytest.raises(RuntimeError, match="stop after mirror capture"):
        live_sim_eval.run_trajectory(
            session=FakeSession(),
            policy=object(),
            args=SimpleNamespace(),
            task_name="hot_dog_setup",
            composite_task="HotDogSetup",
            trajectory=original,
            frames_dir=None,
        )

    assert captured["trajectory"] is original


def test_missing_frames_do_not_mask_harness_error(tmp_path):
    firsts = {}
    _finalize_first_recording(
        frames_dir=tmp_path / "missing",
        output_dir=tmp_path,
        task_name="hot_dog_setup",
        success=False,
        firsts=firsts,
        record_firsts=True,
        save_frames=False,
        record_fps=2,
    )
    assert firsts == {}


def test_resume_retries_and_metrics_supersede_harness_error(tmp_path):
    error = {
        "task_name": "hot_dog_setup",
        "trajectory_id": "traj_000015",
        "termination": "harness_error",
        "native_success": None,
        "fsm_goal_satisfied": None,
    }
    success = {
        "task_name": "hot_dog_setup",
        "trajectory_id": "traj_000015",
        "termination": "budget_exhausted",
        "native_success": False,
        "fsm_goal_satisfied": False,
    }
    key = ("hot_dog_setup", "traj_000015")
    assert _completed_trajectory_keys([error]) == set()
    assert _completed_trajectory_keys([error, success]) == {key}

    results = tmp_path / "live_sim_trajectories.jsonl"
    results.write_text(
        json.dumps(error) + "\n" + json.dumps(success) + "\n",
        encoding="utf-8",
    )
    _write_metrics(results, tmp_path)
    metrics = json.loads((tmp_path / "live_sim_metrics.json").read_text())
    assert metrics["num_trajectories"] == 1
    assert metrics["num_jsonl_records"] == 2
    assert metrics["num_superseded_records"] == 1
    assert metrics["harness_error_rate"] == 0.0


def test_metrics_include_rejections_partial_goal_and_turns(tmp_path):
    result = {
        "native_success": True,
        "fsm_goal_satisfied": True,
        "declared_complete": True,
        "termination": "task_complete_declared",
        "steps_used": 4,
        "expert_steps": 2,
        "partial_goal_fraction": 1.0,
        "step_efficiency_ratio": 2.0,
        "rejected_steps": 1,
        "agent_turns": ["agent_0", "agent_0", "agent_1"],
    }
    results = tmp_path / "live_sim_trajectories.jsonl"
    results.write_text(json.dumps(result) + "\n", encoding="utf-8")
    _write_metrics(results, tmp_path)
    metrics = json.loads((tmp_path / "live_sim_metrics.json").read_text())
    assert metrics["rejection_rate"] == pytest.approx(0.25)
    assert metrics["mean_partial_goal_fraction"] == 1.0
    assert metrics["same_agent_transition_rate"] == pytest.approx(0.5)
    assert metrics["same_agent_run_lengths"] == {"1": 1, "2": 1}


def test_divergence_analysis_ranks_noncontiguous_step_indices(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "sample_id": "task/traj/step_2", "task_name": "task",
            "trajectory_id": "traj", "step_index": 2,
            "target_tool_call": {"name": "communicate"},
            "exact_action_step_match": False,
        },
        {
            "sample_id": "task/traj/step_7", "task_name": "task",
            "trajectory_id": "traj", "step_index": 7,
            "target_tool_call": {"name": "pick_up_object"},
            "exact_action_step_match": True,
        },
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    judge = tmp_path / "judge.jsonl"
    judge.write_text(json.dumps({"sample_id": rows[0]["sample_id"], "equivalent": False}) + "\n")
    output = analyze(predictions, judge)
    first = next(row for row in output if row["row_type"] == "first_divergence_position")
    assert first["key"] == 0
    positions = [row["key"] for row in output if row["row_type"] == "position_accuracy"]
    assert positions == [0, 1]


def test_vllm_policy_preserves_prompt_and_image_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    feature = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "prompt"},
                ],
            },
            {"role": "assistant", "content": ""},
        ],
        "image_paths": [str(first), str(second)],
    }

    messages = live_sim_eval.VllmPolicy._request_messages(feature)

    assert [message["role"] for message in messages] == ["system", "user"]
    user_content = messages[1]["content"]
    assert [item["type"] for item in user_content] == [
        "image_url",
        "image_url",
        "text",
    ]
    assert user_content[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert user_content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert user_content[2]["text"] == "prompt"


def test_vllm_policy_posts_generation_contract_and_rebuilds_tool_call(
    monkeypatch, tmp_path
):
    image = tmp_path / "view.png"
    image.write_bytes(b"view")
    captured = {}
    response_payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "communicate",
                                "arguments": json.dumps(
                                    {
                                        "agent": "agent_0",
                                        "to": "agent_1",
                                        "message": "ready",
                                    }
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(live_sim_eval.urllib_request, "urlopen", fake_urlopen)
    policy = live_sim_eval.VllmPolicy(
        SimpleNamespace(
            vllm_base_url="http://127.0.0.1:8000/v1",
            vllm_model="robocasa-v2",
            vllm_api_key=None,
            vllm_request_timeout=12.0,
            max_new_tokens=256,
            seed=42,
        )
    )
    decoded = policy.generate(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "system"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "prompt"},
                    ],
                },
                {"role": "assistant", "content": ""},
            ],
            "image_paths": [str(image)],
            "tool_schemas": [
                {
                    "type": "function",
                    "function": {
                        "name": "communicate",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["timeout"] == 12.0
    assert captured["payload"]["model"] == "robocasa-v2"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["seed"] == 42
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    parsed = live_sim_eval.parse_first_qwen_tool_call(decoded)
    assert parsed == {
        "name": "communicate",
        "arguments": {
            "agent": "agent_0",
            "to": "agent_1",
            "message": "ready",
        },
    }


def test_model_policy_dispatch_uses_generate_interface():
    class ModelPolicy:
        def generate(self, _feature):
            return ""

    class StepPolicy:
        def next_step(self):
            return None

    assert live_sim_eval._uses_model_generation(ModelPolicy())
    assert live_sim_eval._uses_model_generation(
        object.__new__(live_sim_eval.VllmPolicy)
    )
    assert not live_sim_eval._uses_model_generation(StepPolicy())
