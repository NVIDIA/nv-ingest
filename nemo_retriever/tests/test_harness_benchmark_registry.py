# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_retriever.common.params.utils import build_embed_option_kwargs
from nemo_retriever.harness.benchmark_registry import (
    get_benchmark,
    get_dataset,
    get_runset,
    VIDORE_V3_EMBED_MODEL,
    VIDORE_V3_PUBLIC_DATASETS,
)
from nemo_retriever.harness.contracts import HarnessRunError
from nemo_retriever.harness.resolution import ALLOWED_OVERRIDE_PATHS, build_ingest_request, resolve_benchmark
from nemo_retriever.harness.runfile import load_runfile
from nemo_retriever.ingest.plan import _build_extract_batch_tuning


BO767_VL_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
BO767_VL_HYBRID_RUNFILES = {
    "bo767_vl_text_image_hybrid_beir.json": "text_image",
    "bo767_vl_image_hybrid_beir.json": "image",
}
VIDORE_V3_DATASET_FACTS = {
    "vidore_v3_computer_science": (2, 1360, 1290),
    "vidore_v3_energy": (41, 2225, 1848),
    "vidore_v3_finance_en": (6, 2942, 1854),
    "vidore_v3_finance_fr": (5, 2384, 1920),
    "vidore_v3_hr": (14, 1110, 1908),
    "vidore_v3_industrial": (27, 5244, 1698),
    "vidore_v3_pharmaceuticals": (52, 2313, 2184),
    "vidore_v3_physics": (42, 1674, 1812),
}
RUNFILES_DIR = Path(__file__).resolve().parents[1] / "harness" / "runfiles"


@pytest.mark.parametrize(("runfile_name", "modality"), BO767_VL_HYBRID_RUNFILES.items())
def test_bo767_vl_hybrid_runfiles_resolve_expected_sweep(runfile_name: str, modality: str) -> None:
    request = load_runfile(RUNFILES_DIR / runfile_name)
    resolved = resolve_benchmark(request.benchmark, mode=request.mode or "local", overrides=request.overrides)

    assert request.name == runfile_name.removesuffix(".json")
    assert request.benchmark == "bo767_beir"
    assert request.mode == "batch"
    assert request.requirements == (
        "files==767",
        "pages==54730",
        "query_count==991",
    )
    embed = resolved["ingest"]["embed"]
    assert {key: value for key, value in embed.items() if key != "batch"} == {
        "embed_model_name": BO767_VL_EMBED_MODEL,
        "embed_modality": modality,
        "embed_granularity": "element",
    }
    assert embed["batch"] == {
        "embed_workers": 3,
        "embed_batch_size": 256,
        "embed_gpus_per_actor": 0.1,
    }
    assert resolved["ingest"]["storage"]["index_mode"] == "hybrid"
    assert resolved["query"]["embed_model_name"] == BO767_VL_EMBED_MODEL
    assert resolved["query"]["retrieval_mode"] == "auto"


def test_actor_pool_mode_overrides_reach_batch_tuning_params(tmp_path: Path) -> None:
    extract_key = "ingest.extract.batch.actor_pool_mode"
    embed_key = "ingest.embed.batch.actor_pool_mode"
    assert extract_key in ALLOWED_OVERRIDE_PATHS
    assert embed_key in ALLOWED_OVERRIDE_PATHS

    resolved = resolve_benchmark(
        "jp20_smoke",
        mode="batch",
        overrides=[f"{extract_key}=elastic", f"{embed_key}=elastic"],
    )
    request = build_ingest_request(resolved, tmp_path / "input.pdf", tmp_path)
    extract_tuning = _build_extract_batch_tuning(request.extract.batch)
    embed_tuning = build_embed_option_kwargs(
        None,
        None,
        embed_actor_pool_mode=request.embed.batch.actor_pool_mode,
    )["batch_tuning"]

    assert extract_tuning is not None
    assert extract_tuning.actor_pool_mode == "elastic"
    assert embed_tuning.actor_pool_mode == "elastic"


@pytest.mark.parametrize(
    "override",
    [
        "ingest.extract.batch.actor_pool_mode=auto",
        "ingest.embed.batch.actor_pool_mode=auto",
    ],
)
def test_actor_pool_mode_overrides_reject_invalid_values(override: str, tmp_path: Path) -> None:
    resolved = resolve_benchmark("jp20_smoke", mode="batch", overrides=[override])

    with pytest.raises(HarnessRunError, match="actor_pool_mode must be 'fixed' or 'elastic'"):
        build_ingest_request(resolved, tmp_path / "input.pdf", tmp_path)


@pytest.mark.parametrize("dataset_name", VIDORE_V3_DATASET_FACTS)
def test_vidore_v3_benchmarks_use_page_level_multimodal_defaults(dataset_name: str) -> None:
    dataset = get_dataset(dataset_name)
    benchmark = get_benchmark(f"{dataset_name}_beir")

    assert dataset.beir_loader == "vidore_hf"
    assert dataset.beir_doc_id_field == "pdf_page"
    assert Path(dataset.path).name == dataset_name
    assert benchmark.dataset == dataset_name
    assert benchmark.ingest["extract"] == {
        "extract_infographics": True,
        "extract_page_as_image": True,
    }
    assert benchmark.ingest["embed"] == {
        "embed_model_name": VIDORE_V3_EMBED_MODEL,
        "embed_modality": "text_image",
        "embed_granularity": "page",
    }
    assert benchmark.query["embed_model_name"] == VIDORE_V3_EMBED_MODEL
    assert benchmark.evaluation["dataset_name"] == dataset_name
    assert benchmark.evaluation["doc_id_field"] == "pdf_page"
    assert ("finance" in benchmark.tags) is ("finance" in VIDORE_V3_PUBLIC_DATASETS[dataset_name].lower())


@pytest.mark.parametrize(("dataset_name", "facts"), VIDORE_V3_DATASET_FACTS.items())
def test_vidore_v3_runfiles_carry_dataset_integrity_gates(dataset_name: str, facts: tuple[int, int, int]) -> None:
    files, pages, queries = facts
    request = load_runfile(RUNFILES_DIR / f"{dataset_name}_beir.json")

    assert request.benchmark == f"{dataset_name}_beir"
    assert request.mode == "batch"
    assert request.requirements == (
        f"files=={files}",
        f"pages=={pages}",
        f"query_count=={queries}",
    )


def test_vidore_v3_all_runset_contains_every_public_dataset() -> None:
    runset = get_runset("vidore_v3_all")

    assert runset.runs == tuple(f"{name}_beir" for name in VIDORE_V3_PUBLIC_DATASETS)
