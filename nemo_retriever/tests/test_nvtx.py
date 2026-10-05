# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the shared GPU inference capture marker."""

from contextlib import contextmanager
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_retriever.common import nvtx


def _record_ranges(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    events: list[tuple[str, ...]] = []

    @contextmanager
    def annotate(label: str, *, color: str):
        assert color == "blue"
        events.append(("push", label))
        try:
            yield
        finally:
            events.append(("pop",))

    monkeypatch.setattr(nvtx, "_nvtx", SimpleNamespace(annotate=annotate))
    return events


@pytest.mark.parametrize("fails", [False, True])
def test_gpu_inference_ranges_preserve_labels_and_balance(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    events = _record_ranges(monkeypatch)

    def inference() -> None:
        with nvtx.gpu_inference_range("NemotronOCRv1", batch_size=8, phase="decode"):
            if fails:
                raise ValueError("inference failed")

    if fails:
        with pytest.raises(ValueError, match="inference failed"):
            inference()
    else:
        inference()
    assert events == [
        ("push", "gpu_inference"),
        ("push", "NemotronOCRv1 | bs=8 | phase=decode"),
        ("pop",),
        ("pop",),
    ]


def test_nvtx_without_torch() -> None:
    """The real NVTX backend works when importing torch is forbidden."""
    src = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import builtins
import importlib.util
import sys

real_import = builtins.__import__

def forbid_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("NVTX instrumentation must not import torch")
    return real_import(name, *args, **kwargs)

builtins.__import__ = forbid_torch

spec = importlib.util.spec_from_file_location("nrl_nvtx", sys.argv[1])
nvtx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nvtx)
gpu_inference_range = nvtx.gpu_inference_range

with gpu_inference_range("test", batch_size=8):
    pass

error = ValueError("original workload error")
try:
    with gpu_inference_range("test"):
        raise error
except ValueError as exc:
    assert exc is error
else:
    raise AssertionError("workload exception was swallowed")
assert "torch" not in sys.modules
"""
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script, str(src / "nemo_retriever" / "common" / "nvtx.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
