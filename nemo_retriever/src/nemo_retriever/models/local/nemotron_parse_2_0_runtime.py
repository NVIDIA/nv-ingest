# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activate the selected Parse checkpoint's documented vLLM compatibility code."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path


@contextmanager
def parse_vllm_runtime(model_path: str, revision: str):
    """Apply the pinned tied-head patch here and in spawned Python workers.

    vLLM 0.25.1 allocates a separate output head, while the Parse 2.0 export
    contains only the tied decoder embeddings. Use the checkpoint-supplied
    patch rather than changing weights or relying on a mutable remote script.
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(repo_id=model_path, revision=revision, allow_patterns=["vllm_tied_patch/sitecustomize.py"])
    )
    patch = snapshot / "vllm_tied_patch/sitecustomize.py"
    if not patch.is_file():
        raise FileNotFoundError(f"Parse 2.0 checkpoint {revision} is missing its vLLM tied-head patch")
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(patch.parent) + (os.pathsep + previous if previous else "")
    try:
        spec = importlib.util.spec_from_file_location("_nrl_parse_2_0_tied_head", patch)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the pinned Parse 2.0 vLLM compatibility patch")
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        yield
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous
