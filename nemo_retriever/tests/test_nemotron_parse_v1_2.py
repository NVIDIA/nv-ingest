# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local vLLM-backed Nemotron Parse model."""

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_dependencies_support_transformers_tokenizers_floor():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    tokenizers = next(Requirement(dependency) for dependency in dependencies if dependency.startswith("tokenizers"))

    assert tokenizers.specifier.contains("0.23.1")
    assert not tokenizers.specifier.contains("0.23.0")


def test_applies_vllm_startup_defaults_before_constructing_llm(monkeypatch):
    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    def assert_startup_defaults(**_kwargs):
        assert os.environ["VLLM_DEEP_GEMM_WARMUP"] == "skip"
        return MagicMock()

    with (
        patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
        patch.object(mod, "configure_global_hf_cache_base"),
        patch.object(mod, "get_hf_revision", return_value="test-revision"),
        patch("vllm.LLM", side_effect=assert_startup_defaults),
        patch("vllm.SamplingParams"),
    ):
        mod.NemotronParseV12()


def _local_wrapper(outputs):
    from nemo_retriever.models.local.nemotron_parse_v1_2 import NemotronParseV12

    with patch.object(NemotronParseV12, "__init__", return_value=None):
        model = NemotronParseV12()

    model._task_prompt = "prompt"
    model._sampling_params = object()
    model.preprocess = MagicMock(side_effect=lambda image: image)
    model._llm = MagicMock()
    model._llm.generate.return_value = outputs
    return model


def test_local_wrapper_retains_finish_reason_without_changing_legacy_text_contract() -> None:
    raw_output = "\n<x_0><y_0>Hello<x_1><y_1><class_Text>\t"
    model = _local_wrapper([SimpleNamespace(outputs=[SimpleNamespace(text=raw_output, finish_reason="length")])])

    assert model.invoke_batch_with_finish_reasons(["image"], task_prompt="prompt") == [(raw_output, "length")]
    assert model.invoke_batch(["image"], task_prompt="prompt") == [raw_output.strip()]


def test_local_wrapper_never_shifts_results_when_a_completion_is_missing() -> None:
    model = _local_wrapper(
        [SimpleNamespace(outputs=[]), SimpleNamespace(outputs=[SimpleNamespace(text="page", finish_reason="stop")])]
    )

    with pytest.raises(IndexError):
        model.invoke_batch(["first", "second"])


def test_async_engine_adds_only_the_engine_defaults_the_offline_llm_applies_itself() -> None:
    from vllm.sampling_params import RequestOutputKind

    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    settings = {}
    for async_engine in (False, True):
        with (
            patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value="test-revision"),
            patch.object(mod, "_AsyncEngine") as engine_class,
            patch("vllm.LLM") as llm_class,
            patch("vllm.SamplingParams") as sampling_class,
        ):
            model = mod.NemotronParseV12(async_engine=async_engine)
        if async_engine:
            llm_class.assert_not_called()
            assert model._llm is engine_class.return_value
            engine_kwargs = engine_class.call_args.args[0]
        else:
            engine_class.assert_not_called()
            engine_kwargs = llm_class.call_args.kwargs
        settings[async_engine] = (engine_kwargs, sampling_class.call_args.kwargs)

    (sync_engine, sync_sampling), (async_engine_kwargs, async_sampling) = settings[False], settings[True]
    assert async_engine_kwargs == {**sync_engine, "seed": 0, "disable_log_stats": True}
    assert async_sampling == {**sync_sampling, "output_kind": RequestOutputKind.FINAL_ONLY}


class _FakeAsyncLLM:
    """Finishes "slow" prompts last, fails "bad" ones, never finishes "hang", and records cancellations."""

    def __init__(self) -> None:
        self.request_ids: list[str] = []
        self.cancelled: list[str] = []
        self.shutdown_calls = 0

    async def generate(self, prompt, sampling_params, request_id):
        import asyncio

        self.request_ids.append(request_id)
        try:
            await asyncio.sleep({"slow": 0.2, "hang": 60}.get(prompt, 0))
        except asyncio.CancelledError:
            self.cancelled.append(prompt)
            raise
        if prompt == "bad":
            raise RuntimeError("engine failed")
        yield SimpleNamespace(prompt=prompt)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _make_async_engine(fake: _FakeAsyncLLM):
    from nemo_retriever.models.local.nemotron_parse_v1_2 import _AsyncEngine

    with (
        patch("vllm.engine.arg_utils.AsyncEngineArgs", side_effect=lambda **kwargs: kwargs),
        patch("vllm.v1.engine.async_llm.AsyncLLM.from_engine_args", return_value=fake),
    ):
        return _AsyncEngine({"model": "test"})


@pytest.fixture
def fake_async_engine():
    fake = _FakeAsyncLLM()
    engine = _make_async_engine(fake)
    yield engine, fake
    engine.close()


def test_async_engine_returns_outputs_in_input_order(fake_async_engine) -> None:
    engine, fake = fake_async_engine

    outputs = engine.generate(["slow", "fast"], sampling_params=None)

    assert [output.prompt for output in outputs] == ["slow", "fast"]
    assert len(set(fake.request_ids)) == 2


def test_async_engine_failure_cancels_the_rest_of_the_batch(fake_async_engine) -> None:
    engine, fake = fake_async_engine

    with pytest.raises(RuntimeError, match="engine failed"):
        engine.generate(["slow", "bad"], sampling_params=None)

    assert fake.cancelled == ["slow"]


def test_async_engine_creation_failure_stops_its_event_loop_thread() -> None:
    import threading

    from nemo_retriever.models.local.nemotron_parse_v1_2 import _AsyncEngine

    def engine_threads() -> set[threading.Thread]:
        return {thread for thread in threading.enumerate() if thread.name == "nemotron-parse-engine"}

    before = engine_threads()
    with (
        patch("vllm.engine.arg_utils.AsyncEngineArgs", side_effect=lambda **kwargs: kwargs),
        patch("vllm.v1.engine.async_llm.AsyncLLM.from_engine_args", side_effect=RuntimeError("no GPU")),
        pytest.raises(RuntimeError, match="no GPU"),
    ):
        _AsyncEngine({"model": "test"})

    assert engine_threads() <= before


def test_async_engine_close_cancels_in_flight_requests_and_shuts_down_once() -> None:
    import concurrent.futures
    import threading
    import time

    fake = _FakeAsyncLLM()
    engine = _make_async_engine(fake)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        in_flight = pool.submit(engine.generate, ["hang"], None)
        deadline = time.monotonic() + 5
        while not fake.request_ids and time.monotonic() < deadline:
            time.sleep(0.01)
        closers = [threading.Thread(target=engine.close) for _ in range(2)]
        for closer in closers:
            closer.start()
        for closer in closers:
            closer.join()
        with pytest.raises(concurrent.futures.CancelledError):
            in_flight.result(timeout=5)

    engine.close()
    assert fake.cancelled == ["hang"]
    assert fake.shutdown_calls == 1
    assert not engine._thread.is_alive()
    assert engine._loop.is_closed()


def test_async_engine_cancels_only_the_output_handler_and_batches_before_shutdown() -> None:
    import asyncio

    fake = _FakeAsyncLLM()
    engine = _make_async_engine(fake)

    async def start_engine_tasks():
        fake.output_handler = asyncio.ensure_future(asyncio.sleep(60))
        return asyncio.ensure_future(asyncio.sleep(60))

    other_engine_task = asyncio.run_coroutine_threadsafe(start_engine_tasks(), engine._loop).result()
    seen_at_shutdown = []
    fake.shutdown = lambda: seen_at_shutdown.append((fake.output_handler.cancelled(), other_engine_task.done()))

    engine.close()

    assert seen_at_shutdown == [(True, False)]


def test_async_engine_shuts_down_when_collected() -> None:
    import gc

    fake = _FakeAsyncLLM()
    engine = _make_async_engine(fake)
    thread = engine._thread

    del engine
    gc.collect()

    assert fake.shutdown_calls == 1
    assert not thread.is_alive()


def test_async_engine_close_from_its_own_loop_thread_does_not_deadlock() -> None:
    fake = _FakeAsyncLLM()
    engine = _make_async_engine(fake)

    engine._loop.call_soon_threadsafe(engine.close)
    engine._thread.join(timeout=5)

    assert not engine._thread.is_alive()
    assert fake.shutdown_calls == 1

