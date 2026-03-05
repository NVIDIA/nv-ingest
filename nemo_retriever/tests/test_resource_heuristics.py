import sys
import types

import pytest

from nemo_retriever.utils import ray_resource_hueristics as rh


def test_read_env_int_and_float_fallback_and_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_TEST_INT", " 7 ")
    monkeypatch.setenv("NEMO_TEST_FLOAT", "0.25")
    assert rh._read_env_int("NEMO_TEST_INT", 3, minimum=0) == 7
    assert rh._read_env_float("NEMO_TEST_FLOAT", 1.0, minimum=0.0) == 0.25

    monkeypatch.setenv("NEMO_TEST_INT", "oops")
    monkeypatch.setenv("NEMO_TEST_FLOAT", "oops")
    assert rh._read_env_int("NEMO_TEST_INT", 3, minimum=0) == 3
    assert rh._read_env_float("NEMO_TEST_FLOAT", 1.0, minimum=0.0) == 1.0

    monkeypatch.setenv("NEMO_TEST_INT", "-5")
    monkeypatch.setenv("NEMO_TEST_FLOAT", "-2.5")
    assert rh._read_env_int("NEMO_TEST_INT", 3, minimum=0) == 0
    assert rh._read_env_float("NEMO_TEST_FLOAT", 1.0, minimum=0.0) == 0.0


def test_detect_local_gpu_count_from_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0, 1,2")
    assert rh._detect_local_gpu_count() == 3

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert rh._detect_local_gpu_count() == 0


def test_detect_local_gpu_count_from_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 4))
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    assert rh._detect_local_gpu_count() == 4


def test_resolve_effective_resources_without_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    out = rh.resolve_effective_resources()
    assert out.source == "local"
    assert out.cpu_count >= 1
    assert out.gpu_count >= 0


def test_resolve_effective_resources_precedence_env_then_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setenv(rh.ENV_BATCH_NUM_CPUS, "11")
    monkeypatch.setenv(rh.ENV_BATCH_NUM_GPUS, "3")

    from_env = rh.resolve_effective_resources()
    assert from_env.cpu_count == 11
    assert from_env.gpu_count == 3

    from_args = rh.resolve_effective_resources(override_cpu_count=13, override_gpu_count=4)
    assert from_args.cpu_count == 13
    assert from_args.gpu_count == 4


def test_resolve_batch_worker_plan_scaled_gpu_allocation() -> None:
    out = rh.resolve_batch_worker_plan(
        override_cpu_count=8,
        override_gpu_count=1,
        concurrent_gpu_stage_count=4,
    )
    assert out.page_elements_workers >= 1
    assert out.detect_workers >= 1
    assert out.embed_workers >= 1
    assert out.page_elements_num_gpus == pytest.approx(0.25)
    assert out.detect_num_gpus == pytest.approx(0.25)
    assert out.embed_num_gpus == pytest.approx(0.25)


def test_resolve_batch_worker_plan_high_overlap_allocation() -> None:
    out = rh.resolve_batch_worker_plan(
        override_cpu_count=8,
        override_gpu_count=2,
        concurrent_gpu_stage_count=3,
    )
    assert out.page_elements_num_gpus == rh.HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS
    assert out.detect_num_gpus == rh.HIGH_OVERLAP_OCR_NUM_GPUS
    assert out.embed_num_gpus == rh.HIGH_OVERLAP_EMBED_NUM_GPUS


def test_resolve_batch_worker_plan_env_heuristics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_BATCH_PAGE_ELEMENTS_PER_GPU", "4")
    monkeypatch.setenv("NEMO_RETRIEVER_BATCH_OCR_PER_GPU", "5")
    monkeypatch.setenv("NEMO_RETRIEVER_BATCH_EMBED_PER_GPU", "6")
    out = rh.resolve_batch_worker_plan(override_cpu_count=8, override_gpu_count=2)
    assert out.page_elements_workers == 8
    assert out.detect_workers == 10
    assert out.embed_workers == 12


def test_format_and_pretty_print_worker_heuristic_summary() -> None:
    out = rh.resolve_batch_worker_plan(
        override_cpu_count=16,
        override_gpu_count=1,
        concurrent_gpu_stage_count=2,
    )
    summary = rh.format_worker_heuristic_summary(out)
    assert "Batch worker heuristic configuration:" in summary
    assert "resources: cpu=16, gpu=1" in summary
    assert "gpu_stage_allocation:" in summary

    printed: list[str] = []
    rh.pretty_print_worker_heuristic_summary(out, print_fn=printed.append)
    assert len(printed) == 1
    assert "final_workers:" in printed[0]
