# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure core ingest imports never load local inference or gRPC dependencies.

Remote-NIM and slim installs omit ``tritonclient``; gRPC code paths import it lazily.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_SLIM_IMPORT_SCRIPT = r"""
import builtins

_real_import = builtins.__import__


def _guard(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch import blocked (core-only contract)", name=name)
    if name == "tritonclient" or (
        isinstance(name, str) and name.startswith("tritonclient.")
    ):
        raise ImportError("tritonclient import blocked (slim / API-only contract)")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guard

from nemo_retriever.models.nim.util import create_inference_client  # noqa: F401
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor  # noqa: F401
from nemo_retriever.graph.executor import ray_dataset_to_pandas
from types import SimpleNamespace
import pandas as pd

frame = pd.DataFrame({"value": [42]})
dataset = SimpleNamespace(iter_batches=lambda **kwargs: iter([frame]))
assert ray_dataset_to_pandas(dataset).equals(frame)

print("slim_imports_ok")
"""


def test_core_imports_do_not_require_tritonclient_at_import_time() -> None:
    """Fresh interpreter: block tritonclient, then import remote-ingest paths."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) + (os.pathsep + prev if prev else "")

    proc = subprocess.run(
        [sys.executable, "-c", _SLIM_IMPORT_SCRIPT],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}\n" f"exit={proc.returncode}"
    assert "slim_imports_ok" in proc.stdout
