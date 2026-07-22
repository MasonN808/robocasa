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
