from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import EmbedParams


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


class _DummyDataset:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict[str, object]]] = []

    def repartition(self, **kwargs):
        self.calls.append(("repartition", None, kwargs))
        return self

    def map_batches(self, fn, **kwargs):
        self.calls.append(("map_batches", fn, kwargs))
        return self


class _DummyRequestedPlan:
    def get_embed_batch_size(self) -> int:
        return 4

    def get_embed_gpus_per_actor(self) -> float:
        return 0.0

    def get_embed_initial_actors(self) -> int:
        return 1

    def get_embed_min_actors(self) -> int:
        return 1

    def get_embed_max_actors(self) -> int:
        return 1


def test_write_observability_snapshot_requires_output_dir() -> None:
    ingestor = object.__new__(BatchIngestor)
    ingestor._rd_dataset = _DummyDataset()
    with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
        ingestor.write_observability_snapshot("", stage_name="extract")


def test_embed_inserts_chunk_manifest_snapshot_stage(monkeypatch) -> None:
    ingestor = object.__new__(BatchIngestor)
    ingestor._rd_dataset = _DummyDataset()
    ingestor._requested_plan = _DummyRequestedPlan()
    ingestor._tasks = []

    params = EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2")

    ingestor.embed(
        params,
        chunk_manifest_output_dir="/tmp/chunks",
        durable_chunk_manifest_dir="/tmp/chunks-archive",
        chunk_manifest_drop_columns=["_image_b64"],
    )

    map_batch_calls = [entry for entry in ingestor._rd_dataset.calls if entry[0] == "map_batches"]
    assert len(map_batch_calls) >= 3

    snapshot_call = map_batch_calls[1]
    fn = snapshot_call[1]
    kwargs = snapshot_call[2]

    assert getattr(fn, "func").__name__ == "write_jsonl_snapshot_batch"
    assert fn.keywords["output_dir"] == "/tmp/chunks"
    assert fn.keywords["stage_name"] == "chunk-manifest"
    assert fn.keywords["durable_output_dir"] == "/tmp/chunks-archive"
    assert fn.keywords["drop_columns"] == ["_image_b64"]
    assert kwargs["batch_format"] == "pandas"
    assert kwargs["num_cpus"] == 1
