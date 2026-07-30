#!/usr/bin/env python
"""Pushes the trained v1.5 reasoning-SFT LoRA adapter to the HF Hub.

Usage:
  python push_v1_5_reasoning_hf.py --adapter-dir <path> --repo-id <user/repo> [--private]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {args.adapter_dir}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.adapter_dir),
        commit_message="Upload v1.5 reasoning-SFT LoRA adapter",
    )
    print(f"Pushed {args.adapter_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
