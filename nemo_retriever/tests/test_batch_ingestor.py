from types import SimpleNamespace

from nemo_retriever.ingest_modes.batch import BatchIngestor


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

    monkeypatch.delenv("NEMO_RETRIEVER_HF_CACHE_DIR", raising=False)
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

    assert captured["runtime_env"] == {"env_vars": {"LOG_LEVEL": "INFO"}}
    assert dummy_ctx.enable_rich_progress_bars is True
    assert dummy_ctx.use_ray_tqdm is False
