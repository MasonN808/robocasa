import json
from pathlib import Path

import pytest

from training.bc_task_vlm.build_43_10_eval_artifact import recorded_checkpoint_epoch


@pytest.mark.parametrize('epoch,expected', [(1.0, 'epoch 1.0'), (0.5003, 'epoch 0.5')])
def test_reads_actual_epoch_despite_directory_label(tmp_path: Path, epoch, expected):
    adapter = tmp_path / 'checkpoint-973'
    adapter.mkdir()
    (adapter / 'trainer_state.json').write_text(json.dumps({'epoch': epoch}))
    split = tmp_path / 'incorrect_ep1p0' / 'train_task_types'
    split.mkdir(parents=True)
    (split / 'parallel_run.json').write_text(json.dumps({
        'identity': {'live_args_template': ['--adapter-path', str(adapter)]}
    }))
    assert recorded_checkpoint_epoch(split) == expected


def test_missing_metadata_does_not_silently_use_folder_label(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        recorded_checkpoint_epoch(tmp_path)
