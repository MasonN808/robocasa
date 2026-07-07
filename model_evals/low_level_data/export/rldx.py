"""Export raw segments for RLDX-1 RoboCasa fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_evals.low_level_data.export.lerobot import export_lerobot

RLDX_CONFIG = '# Generated for low-level RoboCasa LeRobot exports.\n\nimport copy\n\nfrom rldx.configs.data.embodiment_configs import register_modality_config\nfrom rldx.data.embodiment_tags import EmbodimentTag\nfrom rldx.data.types import (\n    ActionConfig,\n    ActionFormat,\n    ActionRepresentation,\n    ActionType,\n    ModalityConfig,\n)\n\nrobocasa_low_level_panda_omron = {\n    "video": ModalityConfig(\n        delta_indices=[0],\n        modality_keys=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],\n    ),\n    "state": ModalityConfig(\n        delta_indices=[0],\n        modality_keys=[\n            "end_effector_position_relative",\n            "end_effector_rotation_relative",\n            "gripper_qpos",\n            "base_position",\n            "base_rotation",\n        ],\n    ),\n    "action": ModalityConfig(\n        delta_indices=list(range(16)),\n        modality_keys=[\n            "end_effector_position",\n            "end_effector_rotation",\n            "gripper_close",\n            "base_motion",\n            "control_mode",\n        ],\n        action_configs=[\n            ActionConfig(rep=ActionRepresentation.DELTA, type=ActionType.EEF, format=ActionFormat.DEFAULT),\n            ActionConfig(rep=ActionRepresentation.DELTA, type=ActionType.EEF, format=ActionFormat.DEFAULT),\n            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n            ActionConfig(rep=ActionRepresentation.DELTA, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n        ],\n    ),\n    "language": ModalityConfig(\n        delta_indices=[0],\n        modality_keys=["annotation.human.task_description"],\n    ),\n}\n\nregister_modality_config(robocasa_low_level_panda_omron, EmbodimentTag.GENERAL_EMBODIMENT)\nregister_modality_config(copy.deepcopy(robocasa_low_level_panda_omron), EmbodimentTag.NEW_EMBODIMENT)\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_rejected", action="store_true")
    args = parser.parse_args()
    output = export_lerobot(args.segments_dir, args.output_dir, accepted_only=not args.include_rejected)
    config_path = output / "rldx_robocasa_low_level_config.py"
    config_path.write_text(RLDX_CONFIG)
    (output / "rldx_train_command.txt").write_text(
        "cd external/rldx-1\n"
        "DATA_DIR=<exported_dataset> uv run torchrun --nproc_per_node=<N_GPUS> rldx/experiment/launch_train.py \\\n"
        "  --base-model-path RLWRLD/RLDX-1-PT \\\n"
        "  --dataset-path <exported_dataset> \\\n"
        f"  --modality-config-path {config_path.resolve()} \\\n"
        "  --embodiment-tag GENERAL_EMBODIMENT \\\n"
        "  --video-length 4 --n-cog-tokens 64 --global-batch-size 64 --max-steps <steps>\n"
    )
    print(f"Exported RLDX-compatible LeRobot dataset: {output}")
    print(f"Generated modality config: {config_path}")


if __name__ == "__main__":
    main()
