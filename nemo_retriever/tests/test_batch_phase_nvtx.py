"""Contract tests for semantic batch NVTX instrumentation."""

from collections.abc import Callable

import pytest

from nemo_retriever.common import nvtx


def _record_ranges(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    events: list[tuple[str, ...]] = []
    monkeypatch.setattr(nvtx._nvtx, "range_push", lambda label: events.append(("push", label)))
    monkeypatch.setattr(nvtx._nvtx, "range_pop", lambda: events.append(("pop",)))
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

    monkeypatch.setattr(nvtx._nvtx, "range_push", unavailable)

    assert nvtx.batch_phase("page_elements.batch")(function)() == 7
    assert calls == 1


def test_unexpected_nvtx_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(_label: str) -> None:
        raise RuntimeError("unexpected profiler failure")

    function: Callable[[], int] = lambda: 7
    monkeypatch.setattr(nvtx._nvtx, "range_push", unavailable)

    with pytest.raises(RuntimeError, match="unexpected profiler failure"):
        nvtx.batch_phase("embedding.batch")(function)()
