"""Export raw segments for GR00T N1.5 RoboCasa fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_evals.low_level_data.export.lerobot import export_lerobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_rejected", action="store_true")
    args = parser.parse_args()
    output = export_lerobot(args.segments_dir, args.output_dir, accepted_only=not args.include_rejected)
    snippet = {
        "path": str(output.resolve()),
        "filter_key": None,
        "data_config": "panda_omron",
        "train_entrypoint": "external/Isaac-GR00T/scripts/gr00t_finetune.py",
    }
    (output / "gr00t_dataset_snippet.json").write_text(json.dumps(snippet, indent=2))
    print(f"Exported GR00T-compatible LeRobot dataset: {output}")
    print("Use data_config=panda_omron and add the path to DATASET_SOUP_REGISTRY for training.")


if __name__ == "__main__":
    main()
