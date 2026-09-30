# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for semantic batch NVTX instrumentation."""

from collections.abc import Callable
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_retriever.common import nvtx


def _record_ranges(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    events: list[tuple[str, ...]] = []
    backend = SimpleNamespace(
        range_push=lambda label: events.append(("push", label)),
        range_pop=lambda: events.append(("pop",)),
    )
    monkeypatch.setattr(nvtx, "_get_nvtx", lambda: backend)
    return events


def test_batch_range_is_balanced_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _record_ranges(monkeypatch)

    @nvtx.batch_phase("embedding.batch")
    def instrumented(value: int) -> int:
        return value + 1

    assert instrumented(41) == 42
    assert events == [("push", "nrl.batch::embedding.batch"), ("pop",)]


def test_batch_range_is_balanced_when_function_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _record_ranges(monkeypatch)

    @nvtx.batch_phase("ocr.batch")
    def instrumented() -> None:
        raise ValueError("original")

    with pytest.raises(ValueError, match="original"):
        instrumented()
    assert events == [("push", "nrl.batch::ocr.batch"), ("pop",)]


def test_cpu_only_nvtx_stub_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def unavailable(_label: str) -> None:
        raise RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")

    def function() -> int:
        nonlocal calls
        calls += 1
        return 7

    monkeypatch.setattr(nvtx, "_get_nvtx", lambda: SimpleNamespace(range_push=unavailable))

    assert nvtx.batch_phase("page_elements.batch")(function)() == 7
    assert calls == 1


def test_unexpected_nvtx_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(_label: str) -> None:
        raise RuntimeError("unexpected profiler failure")

    function: Callable[[], int] = lambda: 7
    monkeypatch.setattr(nvtx, "_get_nvtx", lambda: SimpleNamespace(range_push=unavailable))

    with pytest.raises(RuntimeError, match="unexpected profiler failure"):
        nvtx.batch_phase("embedding.batch")(function)()


def test_nvtx_without_site_packages() -> None:
    """A fresh stdlib-only interpreter has no torch, even in all-extras CI."""
    src = Path(__file__).resolve().parents[1] / "src"
    script = r'''
import importlib.util
import sys

assert importlib.util.find_spec("torch") is None

spec = importlib.util.spec_from_file_location("nvtx", sys.argv[1])
nvtx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nvtx)
batch_phase = nvtx.batch_phase
gpu_inference_range = nvtx.gpu_inference_range

calls = []

@batch_phase("embedding.batch")
def instrumented(value):
    """Original metadata."""
    calls.append(value)
    return value + 1

assert instrumented.__name__ == "instrumented"
assert instrumented.__doc__ == "Original metadata."
with gpu_inference_range("test"):
    assert instrumented(41) == 42
assert calls == [41]

error = ValueError("original workload error")

@batch_phase("ocr.batch")
def failing():
    raise error

try:
    failing()
except ValueError as exc:
    assert exc is error
else:
    raise AssertionError("workload exception was swallowed")
assert "torch" not in sys.modules
'''
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(src / "nemo_retriever" / "common" / "nvtx.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
