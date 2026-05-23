"""Disable optional flash_attn discovery for vLLM when requested.

The cluster environment can contain a flash_attn wheel that is present on disk
but ABI-incompatible with the active PyTorch build. vLLM treats flash_attn as
optional in its rotary embedding path, so hiding it from importlib discovery
lets vLLM use its native fallback instead.
"""

from __future__ import annotations

import importlib.util
import os


if os.environ.get("ROBOCASA_DISABLE_FLASH_ATTN_DISCOVERY") == "1":
    _original_find_spec = importlib.util.find_spec

    def _find_spec_without_flash_attn(
        name: str,
        package: str | None = None,
    ):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return _original_find_spec(name, package)

    importlib.util.find_spec = _find_spec_without_flash_attn
