"""Bounded OCR pools preserve explicit initial resource admission."""

import pytest

from nemo_retriever.common.params import BatchTuningParams, ExtractParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import (
    batch_tuning_to_node_overrides,
    build_graph,
    default_concurrency_node_names,
)
from nemo_retriever.ingest.plan import IngestExtractBatchOptions, _build_extract_batch_tuning


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
