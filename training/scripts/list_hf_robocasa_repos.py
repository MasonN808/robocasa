#!/usr/bin/env python3
"""List RoboCasa sweep dataset repos from Hugging Face.

The staging wrappers accept either comma-separated repo IDs or a newline file.
This helper discovers the dated RoboCasa exports under an HF author and writes
both forms so staging jobs do not need to hard-code every task repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Protocol


DEFAULT_AUTHOR = "DorianAtSchool"
DEFAULT_PREFIX = "robocasa_20260430T030150Z_full_"


class DatasetInfoLike(Protocol):
    id: str


class HfApiLike(Protocol):
    def list_datasets(self, *, author: str) -> Iterable[DatasetInfoLike]:
        ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--author",
        default=DEFAULT_AUTHOR,
        help="Hugging Face user or org that owns the dataset repos.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Dataset repo-name prefix to keep.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail if the discovered repo count does not match this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the newline-delimited repo list.",
    )
    parser.add_argument(
        "--env-output",
        type=Path,
        default=None,
        help=(
            "Path for a shell env file containing HF_DATASET_REPOS=repo1,repo2. "
            "Defaults to <output>.env."
        ),
    )
    return parser.parse_args()


def _repo_id(dataset_info: object) -> str:
    repo_id = getattr(dataset_info, "id", None)
    if repo_id is None:
        repo_id = getattr(dataset_info, "repo_id", None)
    if repo_id is None:
        repo_id = str(dataset_info)
    return str(repo_id)


def filter_repo_ids(
    repo_ids: Iterable[str],
    *,
    author: str,
    prefix: str,
) -> list[str]:
    namespace = f"{author}/"
    filtered = []
    for repo_id in repo_ids:
        if not repo_id.startswith(namespace):
            continue
        repo_name = repo_id.split("/", 1)[1]
        if repo_name.startswith(prefix):
            filtered.append(repo_id)
    return sorted(set(filtered))


def discover_repo_ids(
    *,
    author: str,
    prefix: str,
    api: HfApiLike | None = None,
) -> list[str]:
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Missing optional dependency 'huggingface_hub'. Install "
                "training/bc_task_vlm/requirements.txt before listing HF repos."
            ) from exc
        api = HfApi()

    return filter_repo_ids(
        (_repo_id(dataset_info) for dataset_info in api.list_datasets(author=author)),
        author=author,
        prefix=prefix,
    )


def write_repo_outputs(
    repo_ids: list[str],
    *,
    output_path: Path,
    env_output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(repo_ids) + "\n", encoding="utf-8")
    env_output_path.write_text(
        f"HF_DATASET_REPOS={','.join(repo_ids)}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_ids = discover_repo_ids(author=args.author, prefix=args.prefix)
    if args.expected_count is not None and len(repo_ids) != args.expected_count:
        repo_text = "\n".join(repo_ids)
        raise SystemExit(
            f"Expected {args.expected_count} repos but found {len(repo_ids)}.\n"
            f"Discovered repos:\n{repo_text}"
        )

    env_output = args.env_output or args.output.with_suffix(args.output.suffix + ".env")
    write_repo_outputs(
        repo_ids,
        output_path=args.output,
        env_output_path=env_output,
    )
    print(f"Wrote {len(repo_ids)} repo IDs to {args.output}", flush=True)
    print(f"Wrote comma-separated env file to {env_output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # pragma: no cover - shell pipeline behavior
        sys.exit(1)
