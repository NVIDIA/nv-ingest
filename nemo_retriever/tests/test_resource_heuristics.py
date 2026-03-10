import types

import pytest

from nemo_retriever.utils import ray_resource_hueristics as rh


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def test_resources_equality_and_hash() -> None:
    a = rh.Resources(cpu_count=8, gpu_count=2)
    b = rh.Resources(cpu_count=8, gpu_count=2)
    c = rh.Resources(cpu_count=4, gpu_count=1)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "not a Resources"


def test_resources_str_repr() -> None:
    r = rh.Resources(cpu_count=4, gpu_count=1)
    assert "cpu_count=4" in str(r)
    assert "gpu_count=1" in repr(r)


# ---------------------------------------------------------------------------
# ClusterResources
# ---------------------------------------------------------------------------


def test_cluster_resources_accessors() -> None:
    cr = rh.ClusterResources(
        total_resources=rh.Resources(cpu_count=16, gpu_count=4),
        available_resources=rh.Resources(cpu_count=12, gpu_count=3),
    )

    assert cr.total_cpu_count() == 16
    assert cr.total_gpu_count() == 4
    assert cr.available_cpu_count() == 12
    assert cr.available_gpu_count() == 3


def test_cluster_resources_equality() -> None:
    a = rh.ClusterResources(
        total_resources=rh.Resources(cpu_count=8, gpu_count=2),
        available_resources=rh.Resources(cpu_count=6, gpu_count=2),
    )
    b = rh.ClusterResources(
        total_resources=rh.Resources(cpu_count=8, gpu_count=2),
        available_resources=rh.Resources(cpu_count=6, gpu_count=2),
    )
    c = rh.ClusterResources(
        total_resources=rh.Resources(cpu_count=8, gpu_count=2),
        available_resources=rh.Resources(cpu_count=4, gpu_count=1),
    )

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "not a ClusterResources"


# ---------------------------------------------------------------------------
# gather_cluster_resources
# ---------------------------------------------------------------------------


def test_gather_cluster_resources_success() -> None:
    mock_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 32, "GPU": 4},
        available_resources=lambda: {"CPU": 24, "GPU": 3},
    )

    cr = rh.gather_cluster_resources(mock_ray)

    assert cr.total_cpu_count() == 32
    assert cr.total_gpu_count() == 4
    assert cr.available_cpu_count() == 24
    assert cr.available_gpu_count() == 3


def test_gather_cluster_resources_not_initialized() -> None:
    mock_ray = types.SimpleNamespace(is_initialized=lambda: False)

    with pytest.raises(ValueError, match="Ray is not initialized"):
        rh.gather_cluster_resources(mock_ray)


def test_gather_cluster_resources_missing_keys() -> None:
    mock_ray = types.SimpleNamespace(
        is_initialized=lambda: True,
        cluster_resources=lambda: {"CPU": 8},
        available_resources=lambda: {},
    )

    cr = rh.gather_cluster_resources(mock_ray)

    assert cr.total_gpu_count() == 0
    assert cr.available_cpu_count() == 0
    assert cr.available_gpu_count() == 0


# ---------------------------------------------------------------------------
# resolve_requested_plan — defaults
# ---------------------------------------------------------------------------


def _make_cluster(total_cpu: int = 16, total_gpu: int = 2) -> rh.ClusterResources:
    res = rh.Resources(cpu_count=total_cpu, gpu_count=total_gpu)
    return rh.ClusterResources(total_resources=res, available_resources=res)


def test_resolve_requested_plan_defaults_with_2_gpus() -> None:
    plan = rh.resolve_requested_plan(cluster_resources=_make_cluster(total_gpu=2))

    assert plan.embed_initial_actors == rh.EMBED_INITIAL_ACTORS * 2
    assert plan.embed_min_actors == rh.EMBED_MIN_ACTORS * 2
    assert plan.embed_max_actors == rh.EMBED_MAX_ACTORS * 2
    assert plan.embed_gpus_per_actor == rh.EMBED_GPUS_PER_ACTOR
    assert plan.embed_batch_size == rh.EMBED_BATCH_SIZE

    assert plan.ocr_initial_actors == rh.OCR_INITIAL_ACTORS * 2
    assert plan.ocr_max_actors == rh.OCR_MAX_ACTORS * 2
    assert plan.ocr_gpus_per_actor == rh.OCR_GPUS_PER_ACTOR

    assert plan.page_elements_initial_actors == rh.PAGE_ELEMENTS_INITIAL_ACTORS * 2
    assert plan.page_elements_max_actors == rh.PAGE_ELEMENTS_MAX_ACTORS * 2

    assert plan.pdf_extract_batch_size == rh.PDF_EXTRACT_BATCH_SIZE
    assert plan.pdf_extract_cpus_per_task == rh.PDF_EXTRACT_CPUS_PER_TASK
    assert plan.pdf_extract_tasks == rh.PDF_EXTRACT_TASKS * 2


def test_resolve_requested_plan_defaults_with_1_gpu() -> None:
    plan = rh.resolve_requested_plan(cluster_resources=_make_cluster(total_gpu=1))

    assert plan.embed_initial_actors == rh.EMBED_INITIAL_ACTORS
    assert plan.embed_max_actors == rh.EMBED_MAX_ACTORS
    assert plan.ocr_initial_actors == rh.OCR_INITIAL_ACTORS
    assert plan.page_elements_initial_actors == rh.PAGE_ELEMENTS_INITIAL_ACTORS


# ---------------------------------------------------------------------------
# resolve_requested_plan — overrides
# ---------------------------------------------------------------------------


def test_resolve_requested_plan_overrides() -> None:
    plan = rh.resolve_requested_plan(
        cluster_resources=_make_cluster(total_gpu=2),
        override_embed_max_actors=10,
        override_ocr_gpus_per_actor=0.5,
        override_page_elements_batch_size=64,
        override_pdf_extract_tasks=20,
    )

    assert plan.embed_max_actors == 10
    assert plan.ocr_gpus_per_actor == 0.5
    assert plan.page_elements_batch_size == 64
    assert plan.pdf_extract_tasks == 20


# ---------------------------------------------------------------------------
# resolve_requested_plan — edge cases
# ---------------------------------------------------------------------------


def test_resolve_requested_plan_raises_with_no_gpus() -> None:
    with pytest.raises(ValueError, match="No GPUs available"):
        rh.resolve_requested_plan(cluster_resources=_make_cluster(total_gpu=0))


# ---------------------------------------------------------------------------
# RequestedPlan — getters and model behavior
# ---------------------------------------------------------------------------


def test_requested_plan_getters() -> None:
    plan = rh.resolve_requested_plan(cluster_resources=_make_cluster(total_gpu=1))

    assert plan.get_embed_initial_actors() == plan.embed_initial_actors
    assert plan.get_embed_min_actors() == plan.embed_min_actors
    assert plan.get_embed_max_actors() == plan.embed_max_actors
    assert plan.get_embed_gpus_per_actor() == plan.embed_gpus_per_actor
    assert plan.get_embed_batch_size() == plan.embed_batch_size

    assert plan.get_nemotron_parse_initial_actors() == plan.nemotron_parse_initial_actors
    assert plan.get_nemotron_parse_gpus_per_actor() == plan.nemotron_parse_gpus_per_actor

    assert plan.get_ocr_initial_actors() == plan.ocr_initial_actors
    assert plan.get_ocr_batch_size() == plan.ocr_batch_size

    assert plan.get_page_elements_initial_actors() == plan.page_elements_initial_actors
    assert plan.get_page_elements_gpus_per_actor() == plan.page_elements_gpus_per_actor

    assert plan.get_pdf_extract_batch_size() == plan.pdf_extract_batch_size
    assert plan.get_pdf_extract_cpus_per_task() == plan.pdf_extract_cpus_per_task
    assert plan.get_pdf_extract_tasks() == plan.pdf_extract_tasks


def test_requested_plan_equality() -> None:
    cr = _make_cluster(total_gpu=1)
    a = rh.resolve_requested_plan(cluster_resources=cr)
    b = rh.resolve_requested_plan(cluster_resources=cr)

    assert a == b
    assert hash(a) == hash(b)
    assert a != "not a plan"


def test_requested_plan_str_repr() -> None:
    plan = rh.resolve_requested_plan(cluster_resources=_make_cluster(total_gpu=1))
    s = str(plan)
    assert "RequestedPlan" in s
    assert "embed_initial_actors" in s
    assert repr(plan) == s
