import pytest

from data_generation.task_level.generation.raw.exemplar_normalization import (
    ExemplarNormalizationError,
    normalize_exemplar_release_requests,
)


def _candidate(message="Please free the cabinet when finished."):
    return {
        "format": "explicit_blocked_v1",
        "ticks": [
            {
                "tick": 0,
                "agent_0": {
                    "tool": "communicate",
                    "args": {"to": "agent_1", "message": message},
                    "reasoning": "I need the cabinet next.",
                },
                "agent_1": {"tool": "navigate_to_fixture", "args": {"fixture_id": "cab"}, "reasoning": "I will go there."},
            },
            {
                "tick": 1,
                "agent_0": {"tool": "wait_for_signal", "args": {"from": "agent_1", "about": "cab"}, "reasoning": "I will wait."},
                "agent_1": {"tool": "open_fixture_part", "args": {"fixture_id": "cab", "part": "door"}, "reasoning": "I will open it."},
            },
        ],
    }


def test_normalization_only_clarifies_message_text():
    original = _candidate()
    normalized = normalize_exemplar_release_requests(original)
    assert normalized["ticks"][0]["agent_0"]["args"]["message"].endswith(
        'When done, release "cab".'
    )
    assert original["ticks"][0]["agent_0"]["args"]["message"] == "Please free the cabinet when finished."
    before = original.copy()
    before["ticks"] = [dict(row) for row in original["ticks"]]
    assert normalized["ticks"][1] == original["ticks"][1]


def test_normalization_leaves_already_valid_natural_wording_unchanged():
    original = _candidate("Please release cab when finished.")
    assert normalize_exemplar_release_requests(original) == original


def test_normalization_refuses_to_invent_missing_communication_tick():
    original = _candidate()
    original["ticks"][0]["agent_0"] = {
        "tool": "give_space",
        "args": {"fixture_id": "counter"},
        "reasoning": "I will move.",
    }
    with pytest.raises(ExemplarNormalizationError, match="does not immediately follow"):
        normalize_exemplar_release_requests(original)


def test_normalization_inserts_only_logically_implied_blocked_markers():
    original = _candidate("Please release cab when finished.")
    original["ticks"].append(
        {
            "tick": 2,
            "agent_1": {
                "tool": "communicate",
                "args": {"to": "agent_0", "message": "Released.", "releases": "cab"},
                "reasoning": "I will release agent_0.",
            },
        }
    )
    normalized = normalize_exemplar_release_requests(original)
    assert normalized["ticks"][2]["agent_0"] == {"state": "blocked"}


def test_normalization_refuses_an_omitted_unblocked_agent():
    original = _candidate()
    del original["ticks"][0]["agent_1"]
    with pytest.raises(ExemplarNormalizationError, match="omits unblocked agent_1"):
        normalize_exemplar_release_requests(original)
