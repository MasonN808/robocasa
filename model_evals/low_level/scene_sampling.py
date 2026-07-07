"""Official-style scene sampling for low-level trajectory evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from robocasa.models.scenes.scene_registry import unpack_layout_ids, unpack_style_ids


@dataclass(frozen=True)
class SampledScene:
    layout: int
    style: int
    seed: int
    split: str
    scene_seed: int
    episode_index: int
    sampling: str

    def to_dict(self) -> dict:
        return {
            "layout": int(self.layout),
            "style": int(self.style),
            "seed": int(self.seed),
            "split": self.split,
            "scene_seed": int(self.scene_seed),
            "episode_index": int(self.episode_index),
            "sampling": self.sampling,
        }


def scene_pool_for_split(split: str) -> list[tuple[int, int]]:
    if split == "target":
        return [(i, i) for i in range(1, 11)]
    if split == "pretrain":
        layout_ids = unpack_layout_ids(-2)
        style_ids = unpack_style_ids(-2)
        return [(int(layout), int(style)) for layout in layout_ids for style in style_ids]
    if split == "all":
        layout_ids = unpack_layout_ids(-3)
        style_ids = unpack_style_ids(-3)
        return [(int(layout), int(style)) for layout in layout_ids for style in style_ids]
    raise ValueError(f"Unsupported scene split {split!r}")


def sample_official_scene(split: str = "pretrain", scene_seed: int = 7, episode_index: int = 0) -> SampledScene:
    pool = scene_pool_for_split(split)
    if not pool:
        raise ValueError(f"No scenes available for split {split!r}")
    rng = np.random.default_rng(int(scene_seed))
    choice_idx = 0
    for _ in range(int(episode_index) + 1):
        choice_idx = int(rng.choice(len(pool)))
    layout, style = pool[choice_idx]
    return SampledScene(
        layout=int(layout),
        style=int(style),
        seed=int(scene_seed) + int(episode_index),
        split=split,
        scene_seed=int(scene_seed),
        episode_index=int(episode_index),
        sampling="official_split_rng",
    )


def sample_official_scenes(split: str, scene_seed: int, episode_indices: Iterable[int]) -> list[SampledScene]:
    return [sample_official_scene(split=split, scene_seed=scene_seed, episode_index=i) for i in episode_indices]
