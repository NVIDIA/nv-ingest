# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded OCR pools preserve explicit initial resource admission."""

import pandas as pd
import pytest

from nemo_retriever.common.params import BatchTuningParams, ExtractParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import (
    batch_tuning_to_node_overrides,
    build_graph,
    default_concurrency_node_names,
    require_pdf_graph_for_bounded_ocr,
)
from nemo_retriever.ingest.plan import IngestExtractBatchOptions, _build_extract_batch_tuning
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor


@pytest.mark.parametrize(
    "fields",
    [
        {"ocr_min_workers": 2},
        {"ocr_min_workers": 2, "ocr_initial_workers": 2},
        {"ocr_min_workers": 0, "ocr_initial_workers": 2, "ocr_max_workers": 4},
        {"ocr_min_workers": 3, "ocr_initial_workers": 2, "ocr_max_workers": 4},
        {"ocr_min_workers": 2, "ocr_initial_workers": 5, "ocr_max_workers": 4},
        {"ocr_min_workers": True, "ocr_initial_workers": 2, "ocr_max_workers": 4},
        {"ocr_min_workers": 1.5, "ocr_initial_workers": 2, "ocr_max_workers": 4},
        {"ocr_min_workers": 2, "ocr_initial_workers": 2, "ocr_max_workers": 4, "ocr_workers": 2},
    ],
)
def test_invalid_pool_rejected_by_plan_conversion_and_params(fields):
    with pytest.raises(ValueError):
        BatchTuningParams(**fields)
    with pytest.raises(ValueError):
        _build_extract_batch_tuning(IngestExtractBatchOptions(**fields))


@pytest.mark.parametrize("mode", ["auto", "image", "html", "audio", "text", "video"])
def test_bounded_ocr_rejected_without_dedicated_pdf_graph(mode):
    params = ExtractParams(
        batch_tuning=BatchTuningParams(
            ocr_min_workers=2,
            ocr_initial_workers=2,
            ocr_max_workers=4,
        )
    )
    with pytest.raises(ValueError, match="dedicated PDF batch extraction graph"):
        require_pdf_graph_for_bounded_ocr(params, (mode,))


@pytest.mark.parametrize("version,actor", [("v1", "OCRActor"), ("v2", "OCRActor")])
def test_pool_survives_runtime_and_preflight(version, actor):
    tuning = _build_extract_batch_tuning(
        IngestExtractBatchOptions(
            ocr_min_workers=2,
            ocr_initial_workers=3,
            ocr_max_workers=6,
            ocr_cpus_per_actor=0.5,
            ocr_gpus_per_actor=0.5,
        )
    )
    params = ExtractParams(ocr_version=version, batch_tuning=tuning)
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=32, gpu_count=2),
        available_resources=Resources(cpu_count=32, gpu_count=2),
    )
    overrides = batch_tuning_to_node_overrides(params, None, cluster_resources=cluster)
    assert overrides[actor]["concurrency"] == (2, 6, 3)
    assert overrides[actor]["num_cpus"] == 0.5
    assert overrides[actor]["num_gpus"] == 0.5
    automatic = default_concurrency_node_names(params, None, None, None)
    assert actor not in automatic
    graph = build_graph(extract_params=params, stage_order=())
    executor = RayDataExecutor(graph, node_overrides=overrides, auto_concurrency_nodes=automatic)
    # Three initial OCR actors fit; six maximum actors would not. Preflight must
    # preserve the explicit pool instead of clipping its maximum to startup capacity.
    executor._preflight_resources(executor._linearize(graph), 32, 2)
    assert executor._node_overrides[actor]["concurrency"] == (2, 6, 3)
    with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
        executor._preflight_resources(executor._linearize(graph), 32, 1)


def test_pool_bounds_reach_ray_map_batches(monkeypatch):
    import sys
    from types import SimpleNamespace

    class FakeDataContext:
        enable_rich_progress_bars = False
        use_ray_tqdm = True
        batch_to_block_arrow_format = True
        enable_tensor_extension_casting = True

        @classmethod
        def get_current(cls):
            return cls()

    calls = []

    class FakeDataset:
        def __init__(self):
            self.context = FakeDataContext()

        @classmethod
        def copy(cls, _dataset, _deep_copy=False):
            return cls()

        def repartition(self, **_kwargs):
            return self

        def map_batches(self, _operator_class, **kwargs):
            if kwargs["fn_constructor_kwargs"]["operator_class"].__name__ == "OCRActor":
                calls.append(kwargs)
            return self

    fake_ray_data = SimpleNamespace(Dataset=FakeDataset, DataContext=FakeDataContext)
    fake_ray = SimpleNamespace(data=fake_ray_data)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
    monkeypatch.setattr("nemo_retriever.graph.executor.ensure_local_ray_runtime", lambda _address: fake_ray)
    monkeypatch.setattr("nemo_retriever.graph.executor.resolve_graph", lambda graph, _cluster: graph)

    params = ExtractParams(batch_tuning=BatchTuningParams(ocr_min_workers=2, ocr_initial_workers=3, ocr_max_workers=6))
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=32, gpu_count=2),
        available_resources=Resources(cpu_count=32, gpu_count=2),
    )
    graph = build_graph(extract_params=params, stage_order=())
    executor = RayDataExecutor(
        graph,
        node_overrides=batch_tuning_to_node_overrides(params, None, cluster_resources=cluster),
    )
    executor._resources_preflight_complete = True
    executor._preflight_cluster_resources = cluster
    executor.build_dataset(FakeDataset())

    assert len(calls) == 1
    assert calls[0]["concurrency"] == (2, 6, 3)


def test_fixed_and_unspecified_ocr_keep_existing_behavior():
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=32, gpu_count=4),
        available_resources=Resources(cpu_count=32, gpu_count=4),
    )
    fixed = ExtractParams(batch_tuning=BatchTuningParams(ocr_workers=2))
    overrides = batch_tuning_to_node_overrides(fixed, None, cluster_resources=cluster)
    assert overrides["OCRActor"]["concurrency"] == 2
    assert "OCRActor" not in default_concurrency_node_names(fixed, None, None, None)
    assert "OCRActor" in default_concurrency_node_names(ExtractParams(), None, None, None)


def _run_batch_branches(monkeypatch, paths, params):
    branch_overrides = []

    class FakeCluster:
        def available_cpu_count(self):
            return 32

        def total_cpu_count(self):
            return 32

        def available_gpu_count(self):
            return 1

        def total_gpu_count(self):
            return 1

    class FakeDataset:
        def union(self, _other):
            return self

    class FakeExecutor:
        def __init__(self, _graph, **kwargs):
            branch_overrides.append(kwargs["node_overrides"])

        def build_dataset(self, _data, **_kwargs):
            return FakeDataset()

        def ingest(self, _data, **_kwargs):
            return pd.DataFrame({"done": [True]})

    monkeypatch.setattr(GraphIngestor, "_ensure_batch_runtime", lambda self: (None, FakeCluster()))
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.RayDataExecutor", FakeExecutor)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.preflight_executors", lambda *_a, **_k: None)
    monkeypatch.setattr("nemo_retriever.ingestor.branch_extraction.normalize_ray_branch_datasets", lambda ds: ds)
    GraphIngestor(run_mode="batch").files([str(path) for path in paths]).extract(params).ingest()
    return branch_overrides


@pytest.mark.parametrize("other", ["notes.txt", "scan.png"])
def test_mixed_batch_applies_bounded_ocr_to_pdf_branch(monkeypatch, tmp_path, other):
    pdf, other_path = tmp_path / "report.pdf", tmp_path / other
    pdf.write_bytes(b"pdf")
    other_path.write_bytes(b"data")
    params = ExtractParams(batch_tuning=BatchTuningParams(ocr_min_workers=2, ocr_initial_workers=2, ocr_max_workers=4))

    branch_overrides = _run_batch_branches(monkeypatch, [pdf, other_path], params)

    assert branch_overrides[0]["OCRActor"]["concurrency"] == (2, 4, 2)


def test_mixed_batch_without_pdf_rejects_bounded_ocr(monkeypatch, tmp_path):
    image, text = tmp_path / "scan.png", tmp_path / "notes.txt"
    image.write_bytes(b"png")
    text.write_bytes(b"txt")
    params = ExtractParams(batch_tuning=BatchTuningParams(ocr_min_workers=2, ocr_initial_workers=2, ocr_max_workers=4))

    with pytest.raises(ValueError, match="dedicated PDF batch extraction graph"):
        _run_batch_branches(monkeypatch, [image, text], params)
