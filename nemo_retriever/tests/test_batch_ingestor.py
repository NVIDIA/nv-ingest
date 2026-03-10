from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor, _batch_tuning_to_requested_plan_overrides


class _DummyClusterResources:
    def total_cpu_count(self) -> int:
        return 4

    def total_gpu_count(self) -> int:
        return 0

    def available_cpu_count(self) -> int:
        return 4

    def available_gpu_count(self) -> int:
        return 0


def test_batch_ingestor_filters_none_runtime_env_vars(monkeypatch) -> None:
    captured: dict[str, object] = {}
    dummy_ctx = SimpleNamespace(enable_rich_progress_bars=False, use_ray_tqdm=True)

    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_hf_cache_dir",
        lambda: "/tmp/hf-cache",
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.ray.init",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.rd.DataContext.get_current",
        lambda: dummy_ctx,
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.gather_cluster_resources",
        lambda _ray: _DummyClusterResources(),
    )
    monkeypatch.setattr(
        "nemo_retriever.ingest_modes.batch.resolve_requested_plan",
        lambda cluster_resources: {"plan": "dummy"},
    )

    BatchIngestor(documents=[])

    assert captured["runtime_env"] == {
        "env_vars": {
            "LOG_LEVEL": "INFO",
            "NEMO_RETRIEVER_HF_CACHE_DIR": "/tmp/hf-cache",
        }
    }
    assert dummy_ctx.enable_rich_progress_bars is True
    assert dummy_ctx.use_ray_tqdm is False


def test_batch_tuning_to_requested_plan_overrides_maps_fixed_and_auto_values() -> None:
    overrides = _batch_tuning_to_requested_plan_overrides(
        {
            "pdf_extract_workers": 12,
            "pdf_extract_num_cpus": 2.5,
            "pdf_extract_batch_size": 8,
            "page_elements_batch_size": 16,
            "page_elements_workers": 6,
            "detect_workers": 0,
            "detect_batch_size": 0,
            "embed_workers": 4,
            "embed_batch_size": 128,
            "gpu_page_elements": 0.2,
            "gpu_ocr": 0.0,
            "gpu_embed": 0.5,
        }
    )

    assert overrides == {
        "override_pdf_extract_tasks": 12,
        "override_pdf_extract_cpus_per_task": 2.5,
        "override_pdf_extract_batch_size": 8,
        "override_page_elements_batch_size": 16,
        "override_page_elements_initial_actors": 6,
        "override_page_elements_min_actors": 6,
        "override_page_elements_max_actors": 6,
        "override_ocr_initial_actors": None,
        "override_ocr_min_actors": None,
        "override_ocr_max_actors": None,
        "override_ocr_batch_size": None,
        "override_embed_initial_actors": 4,
        "override_embed_min_actors": 4,
        "override_embed_max_actors": 4,
        "override_embed_batch_size": 128,
        "override_page_elements_gpus_per_actor": 0.2,
        "override_ocr_gpus_per_actor": None,
        "override_embed_gpus_per_actor": 0.5,
    }
