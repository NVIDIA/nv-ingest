# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-safe registration of the official, pinned Parse 3.5 vLLM plugin."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision

PARSE35_SNAPSHOT_ENV = "NRL_NEMOTRON_PARSE35_SNAPSHOT"


def register_nemotron_parse35_vllm() -> None:
    """Register vendor model, NaViT, and MTP support in the current vLLM process.

    The adapter sets ``NRL_NEMOTRON_PARSE35_SNAPSHOT`` to its downloaded immutable
    snapshot before engine startup. vLLM invokes this entry point in workers too.
    Other vLLM models leave it unset, so importing their plugins does no Parse work.
    No package is installed and no vendor code is copied into NRL.

    Args:
        None. The snapshot path is inherited through the process environment.

    Returns:
        None. Registers the official plugin only when a Parse snapshot is selected.

    Raises:
        RuntimeError: If the snapshot is not the pinned revision, a different
            plugin is already imported, or the vendor runtime guard rejects versions.
        ImportError: If the snapshot plugin or its runtime dependencies cannot load.
        OSError: If the plugin files cannot be read.
    """
    snapshot = os.environ.get(PARSE35_SNAPSHOT_ENV)
    if not snapshot:
        return
    root = Path(snapshot).resolve()
    if root.name != get_hf_revision(NEMOTRON_PARSE35_MODEL):
        raise RuntimeError("Parse 3.5 vLLM plugin must use the pinned checkpoint snapshot.")
    package_path = root / "vllm_tied_patch" / "__init__.py"
    existing = sys.modules.get("vllm_tied_patch")
    if existing is not None:
        if Path(existing.__file__).resolve() != package_path.resolve():
            raise RuntimeError("A different Parse vLLM plugin is already imported; use a fresh process.")
    else:
        spec = importlib.util.spec_from_file_location("vllm_tied_patch", package_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the Parse 3.5 plugin at {package_path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules["vllm_tied_patch"] = package
        spec.loader.exec_module(package)
    plugin = importlib.import_module("vllm_tied_patch.plugin")
    plugin.register()
