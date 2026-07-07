"""Export raw segments for pi0.5/OpenPI RoboCasa fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_evals.low_level_data.export.lerobot import export_lerobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_rejected", action="store_true")
    args = parser.parse_args()
    output = export_lerobot(args.segments_dir, args.output_dir, accepted_only=not args.include_rejected)
    (output / "openpi_commands.txt").write_text(
        "# Add this dataset path to robocasa.utils.dataset_registry.DATASET_SOUP_REGISTRY, then run from external/openpi:\n"
        "uv run python scripts/compute_norm_stats.py --config-name <your_pi05_config>\n"
        "XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 uv run python scripts/train.py <your_pi05_config> --exp-name=<run_name>\n"
    )
    print(f"Exported pi0.5/OpenPI-compatible LeRobot dataset: {output}")


if __name__ == "__main__":
    main()
