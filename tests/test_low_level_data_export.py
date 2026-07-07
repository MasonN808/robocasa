import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from model_evals.low_level_data.recorder import SegmentWriter
from model_evals.low_level_data.schema import hdf5_action_to_lerobot, lerobot_action_to_hdf5


def _obs(i=0):
    return {
        "state.base_position": np.array([i, 0, 0], dtype=np.float32),
        "state.base_rotation": np.array([0, 0, 0, 1], dtype=np.float32),
        "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "state.end_effector_rotation_relative": np.array([0, 0, 0, 1], dtype=np.float32),
        "state.gripper_qpos": np.array([0.04, 0.04], dtype=np.float32),
    }


def test_action_order_round_trip():
    hdf5 = np.arange(12, dtype=np.float32)
    lerobot = hdf5_action_to_lerobot(hdf5)
    assert lerobot.tolist() == [7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(lerobot_action_to_hdf5(lerobot), hdf5)


def test_segment_writer_saves_arrays_and_metadata(tmp_path):
    writer = SegmentWriter(tmp_path)
    segment = writer.write(
        metadata={"task": "DeliverStraw", "prompt": "pick up the straw"},
        observations=[_obs(0), _obs(1)],
        actions_hdf5=np.ones((1, 12), dtype=np.float32),
        accepted=True,
    )
    assert (segment / "metadata.json").exists()
    assert np.load(segment / "arrays" / "states.npy").shape == (2, 16)
    assert np.load(segment / "arrays" / "actions_lerobot_order.npy").shape == (1, 12)
    metadata = json.loads((segment / "metadata.json").read_text())
    assert metadata["accepted"] is True


def test_lerobot_export_metadata(tmp_path):
    pytest.importorskip("pyarrow")
    from model_evals.low_level_data.export.lerobot import export_lerobot

    segments = tmp_path / "segments"
    writer = SegmentWriter(segments)
    writer.write(
        metadata={"task": "DeliverStraw", "prompt": "pick up the straw", "accepted_by_operator": True},
        observations=[_obs(0), _obs(1)],
        actions_hdf5=np.ones((1, 12), dtype=np.float32),
        accepted=True,
    )
    output = export_lerobot(segments, tmp_path / "lerobot")
    assert (output / "meta" / "modality.json").exists()
    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [16]
    assert info["features"]["action"]["shape"] == [12]
