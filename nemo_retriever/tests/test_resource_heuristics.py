import sys
import types
from pathlib import Path

import pytest
import yaml

from nemo_retriever.ingest_modes import resource_heuristics as rh


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


def test_get_cluster_or_local_resources_without_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    out = rh.get_cluster_or_local_resources()
    assert out.source == "local"
    assert out.cpu_count >= 1
    assert out.gpu_count >= 0


def test_resolve_worker_heuristic_high_cpu_with_overrides() -> None:
    out = rh.resolve_worker_heuristic(
        num_cpus=128,
        num_gpus=2,
        page_elements_workers=9,
        detect_workers=8,
        embed_workers=7,
        gpu_stage_count=3,
    )
    assert out.profile_name == "high_cpu"
    assert out.page_elements_workers == 9
    assert out.detect_workers == 8
    assert out.embed_workers == 7
    assert out.page_elements_num_gpus == rh.HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS
    assert out.detect_num_gpus == rh.HIGH_OVERLAP_OCR_NUM_GPUS
    assert out.embed_num_gpus == rh.HIGH_OVERLAP_EMBED_NUM_GPUS


def test_resolve_worker_heuristic_low_cpu_scaled_gpu_allocation() -> None:
    out = rh.resolve_worker_heuristic(num_cpus=8, num_gpus=1, gpu_stage_count=4)
    assert out.profile_name == "low_cpu"
    assert out.page_elements_workers >= 1
    assert out.detect_workers >= 1
    assert out.embed_workers >= 1
    assert out.page_elements_num_gpus == pytest.approx(0.25)
    assert out.detect_num_gpus == pytest.approx(0.25)
    assert out.embed_num_gpus == pytest.approx(0.25)


def test_format_and_pretty_print_worker_heuristic_summary() -> None:
    out = rh.resolve_worker_heuristic(num_cpus=16, num_gpus=1, gpu_stage_count=2)
    summary = rh.format_worker_heuristic_summary(out)
    assert "Batch worker heuristic configuration:" in summary
    assert "resources: cpu=16, gpu=1" in summary
    assert "gpu_stage_allocation:" in summary

    printed: list[str] = []
    rh.pretty_print_worker_heuristic_summary(out, print_fn=printed.append)
    assert len(printed) == 1
    assert "final_workers:" in printed[0]


def test_get_resources_precedence_autodetect_then_config_then_env_then_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("resources:\n  cpu_count: 8\n  gpu_count: 2\n", encoding="utf-8")

    from_config = rh.get_cluster_or_local_resources(config_path=config_path)
    assert from_config.cpu_count == 8
    assert from_config.gpu_count == 2

    monkeypatch.setenv(rh.ENV_BATCH_NUM_CPUS, "11")
    monkeypatch.setenv(rh.ENV_BATCH_NUM_GPUS, "3")
    from_env = rh.get_cluster_or_local_resources(config_path=config_path)
    assert from_env.cpu_count == 11
    assert from_env.gpu_count == 3

    from_args = rh.get_cluster_or_local_resources(config_path=config_path, num_cpus=13, num_gpus=4)
    assert from_args.cpu_count == 13
    assert from_args.gpu_count == 4


def test_resolve_worker_heuristic_uses_config_and_env_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "heuristics:\n"
            "  cpu_threshold_workers: 4\n"
            "  max_gpu_per_stage: 0.2\n"
            "  high_overlap_page_elements_num_gpus: 0.4\n"
            "  high_overlap_ocr_num_gpus: 0.9\n"
            "  high_overlap_embed_num_gpus: 0.3\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NEMO_RETRIEVER_BATCH_MAX_GPU_PER_STAGE", "0.6")

    out = rh.resolve_worker_heuristic(num_cpus=8, num_gpus=2, gpu_stage_count=4, config_path=config_path)
    assert out.profile_name == "high_cpu"
    assert out.cpu_threshold_workers == 4
    # 2 GPUs / 4 stages = 0.5, capped by env max_gpu_per_stage=0.6.
    assert out.page_elements_num_gpus == pytest.approx(0.5)

    overlap = rh.resolve_worker_heuristic(num_cpus=8, num_gpus=2, gpu_stage_count=3, config_path=config_path)
    assert overlap.page_elements_num_gpus == pytest.approx(0.4)
    assert overlap.detect_num_gpus == pytest.approx(0.9)
    assert overlap.embed_num_gpus == pytest.approx(0.3)


def test_freeze_writes_default_home_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setitem(sys.modules, "ray", None)

    out_path = rh.freeze()
    assert out_path == home / ".nemo-retriever" / "config.yaml"
    assert out_path.exists()

    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "resources" in payload
    assert "heuristics" in payload
    assert "cpu_count" in payload["resources"]
    assert "cpu_threshold_workers" in payload["heuristics"]


def test_get_resources_precedence_partial_args_only_override_provided_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("resources:\n  cpu_count: 10\n  gpu_count: 2\n", encoding="utf-8")
    monkeypatch.setenv(rh.ENV_BATCH_NUM_CPUS, "12")
    monkeypatch.setenv(rh.ENV_BATCH_NUM_GPUS, "4")

    only_cpu_arg = rh.get_cluster_or_local_resources(config_path=config_path, num_cpus=20)
    assert only_cpu_arg.cpu_count == 20
    # GPU still comes from env (higher precedence than config).
    assert only_cpu_arg.gpu_count == 4

    only_gpu_arg = rh.get_cluster_or_local_resources(config_path=config_path, num_gpus=6)
    # CPU still comes from env (higher precedence than config).
    assert only_gpu_arg.cpu_count == 12
    assert only_gpu_arg.gpu_count == 6


def test_resolve_worker_heuristic_resource_inputs_follow_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ray", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "resources:\n  cpu_count: 8\n  gpu_count: 1\nheuristics:\n  cpu_threshold_workers: 6\n",
        encoding="utf-8",
    )

    # No env, no explicit args -> resources come from config.
    from_config = rh.resolve_worker_heuristic(config_path=config_path, gpu_stage_count=2)
    assert from_config.cpu_count == 8
    assert from_config.gpu_count == 1
    assert from_config.profile_name == "high_cpu"

    # Env overrides config.
    monkeypatch.setenv(rh.ENV_BATCH_NUM_CPUS, "4")
    monkeypatch.setenv(rh.ENV_BATCH_NUM_GPUS, "3")
    from_env = rh.resolve_worker_heuristic(config_path=config_path, gpu_stage_count=3)
    assert from_env.cpu_count == 4
    assert from_env.gpu_count == 3
    assert from_env.profile_name == "low_cpu"

    # Explicit args override env/config/autodetect.
    from_args = rh.resolve_worker_heuristic(
        config_path=config_path,
        num_cpus=16,
        num_gpus=2,
        gpu_stage_count=3,
    )
    assert from_args.cpu_count == 16
    assert from_args.gpu_count == 2
    assert from_args.profile_name == "high_cpu"
