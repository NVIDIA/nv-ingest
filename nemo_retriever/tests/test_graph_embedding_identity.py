# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph actor/index identities must agree after local/remote variant selection."""

from unittest.mock import MagicMock

import pytest

from nemo_retriever.common.params import EmbedParams, VdbUploadParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources, Resources
from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph
from nemo_retriever.graph.operator_resolution import resolve_graph
from nemo_retriever.models import (
    NEMOTRON_3_EMBED_BF16_MODEL,
    NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_35_EMBED_BF16_MODEL,
    NEMOTRON_35_EMBED_NVFP4_MODEL,
)
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.operators.vdb import IngestVdbOperator


def resolved_stages(tmp_path, params, *, gpu_count=1, vdb_kwargs=None, resources=None):
    graph = build_post_extract_graph(
        embed_params=params,
        vdb_upload_params=VdbUploadParams(vdb_kwargs={"uri": str(tmp_path), **(vdb_kwargs or {})}),
        stage_order=("embed",),
        reshape_content_before_embed=False,
    )
    resolved = resolve_graph(graph, resources or Resources(cpu_count=8, gpu_count=gpu_count))
    embed_node = resolved.roots[0]
    sink_node = embed_node.children[0]
    assert sink_node.operator_class is IngestVdbOperator
    return embed_node.operator_kwargs["params"], sink_node.operator_kwargs


@pytest.mark.parametrize(
    "backend,native_nvfp4,expected",
    [
        ("vllm", True, NEMOTRON_35_EMBED_NVFP4_MODEL),
        ("vllm", False, NEMOTRON_35_EMBED_BF16_MODEL),
        ("hf", True, NEMOTRON_35_EMBED_BF16_MODEL),
    ],
)
def test_graph_default_actor_and_sink_share_checkpoint(tmp_path, monkeypatch, backend, native_nvfp4, expected):
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: native_nvfp4)
    # The truly unspecified Python .embed() path must also be covered.
    params = EmbedParams() if backend == "vllm" else EmbedParams(local_ingest_embed_backend=backend)
    actor_params, sink_kwargs = resolved_stages(tmp_path, params)
    metadata = sink_kwargs["vdb_kwargs"]
    assert actor_params.model_name == actor_params.embed_model_name == metadata["embedding_model_name"] == expected
    assert actor_params.embed_model_revision == metadata["embedding_model_revision"] == get_hf_revision(expected)
    assert metadata["vector_dim"] is None
    assert "_embedding_params" not in sink_kwargs
    assert params.model_name is None and params.embed_model_revision is None
    from nemo_retriever.graph.retriever import Retriever

    query_kwargs = Retriever()._resolve_embed_kwargs(
        metadata["embedding_model_name"], None, metadata["embedding_model_revision"]
    )
    assert query_kwargs["model_name"] == query_kwargs["embed_model_name"] == actor_params.model_name
    assert query_kwargs["embed_model_revision"] == actor_params.embed_model_revision


def test_graph_embed_model_name_takes_precedence_and_pins_hub_once(tmp_path, monkeypatch):
    revision_resolver = MagicMock(return_value="a" * 40)
    monkeypatch.setattr("nemo_retriever.models.embed_model_spec.resolve_embed_model_revision", revision_resolver)
    actor_params, sink_kwargs = resolved_stages(
        tmp_path, EmbedParams(model_name="other/model", embed_model_name="custom/embed")
    )
    assert actor_params.model_name == actor_params.embed_model_name == "custom/embed"
    assert sink_kwargs["vdb_kwargs"]["embedding_model_name"] == "custom/embed"
    assert actor_params.embed_model_revision == sink_kwargs["vdb_kwargs"]["embedding_model_revision"] == "a" * 40
    assert revision_resolver.call_args_list[0].args == ("custom/embed", None)
    revision_resolver.assert_called_once_with("custom/embed", None)


@pytest.mark.parametrize("remote", [False, True])
def test_cpu_and_remote_defaults_remain_n3(tmp_path, remote):
    params = EmbedParams(embedding_endpoint="http://nim:8000/v1" if remote else None)
    actor_params, sink_kwargs = resolved_stages(tmp_path, params, gpu_count=1 if remote else 0)
    metadata = sink_kwargs["vdb_kwargs"]
    assert actor_params.model_name is None
    assert metadata["embedding_model_name"] == "nvidia/nemotron-3-embed-1b"
    assert "embedding_model_revision" not in metadata


@pytest.mark.parametrize("run_mode", ["inprocess", "batch"])
@pytest.mark.parametrize("gpu_count", [0, 1])
@pytest.mark.parametrize("with_embed_options", [False, True])
def test_cli_default_is_resolved_by_selected_actor(tmp_path, monkeypatch, run_mode, gpu_count, with_embed_options):
    from nemo_retriever.ingest.plan import (
        IngestEmbedOptions,
        IngestPlanRequest,
        IngestRuntimeOptions,
        IngestSourceOptions,
        IngestStorageOptions,
        resolve_ingest_plan,
    )
    from nemo_retriever.models.inference.shared import build_embed_kwargs
    from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor
    from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor as BatchEmbedGPUActor

    document = tmp_path / "input.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    driver_probe = MagicMock(return_value=True)
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", driver_probe)
    plan = resolve_ingest_plan(
        IngestPlanRequest(
            source=IngestSourceOptions(documents=[str(document)]),
            runtime=IngestRuntimeOptions(run_mode=run_mode),
            # Exercise non-model options as well as the entirely omitted embed params.
            embed=IngestEmbedOptions(embed_granularity="page") if with_embed_options else IngestEmbedOptions(),
            storage=IngestStorageOptions(lancedb_uri=str(tmp_path / "index")),
        )
    )
    driver_probe.assert_not_called()
    params = plan.embed_params or EmbedParams()
    assert params.model_name is None
    assert params.embed_model_name is None
    assert "embedding_model_name" not in plan.vdb_params.vdb_kwargs
    assert "embedding_model_revision" not in plan.vdb_params.vdb_kwargs
    resources = Resources(cpu_count=8, gpu_count=gpu_count)
    if run_mode == "batch":
        resources = ClusterResources(total_resources=resources, available_resources=resources)
    graph = build_post_extract_graph(
        embed_params=params,
        vdb_upload_params=plan.vdb_params,
        stage_order=("embed",),
        reshape_content_before_embed=False,
    )
    resolved = resolve_graph(graph, resources)
    embed_node = resolved.roots[0]
    metadata = embed_node.children[0].operator_kwargs["vdb_kwargs"]
    params = embed_node.operator_kwargs["params"]
    if gpu_count:
        expected = NEMOTRON_35_EMBED_BF16_MODEL if run_mode == "batch" else NEMOTRON_35_EMBED_NVFP4_MODEL
        assert embed_node.operator_class is BatchEmbedGPUActor
        assert params.model_name == params.embed_model_name == metadata["embedding_model_name"] == expected
        assert params.embed_model_revision == metadata["embedding_model_revision"] == get_hf_revision(expected)
    else:
        import pandas as pd
        from nemo_retriever.models.inference import runtime
        from nemo_retriever.models.inference.main_text_embed import assign_embedding_columns

        assert embed_node.operator_class is _BatchEmbedCPUActor
        assert metadata["embedding_model_name"] == NEMOTRON_3_EMBED_MODEL
        assert "embedding_model_revision" not in metadata
        # The CPU actor adds the hosted endpoint, then request construction must use N3.
        kwargs = build_embed_kwargs(params)
        kwargs["embedding_endpoint"] = _BatchEmbedCPUActor.DEFAULT_EMBED_INVOKE_URL
        request = MagicMock(
            side_effect=lambda frame, **options: (
                assign_embedding_columns(frame, [[1.0]] * len(frame), [None] * len(frame), output_column="embedding"),
                {},
            )
        )
        monkeypatch.setattr(runtime, "create_text_embeddings_for_df", request)
        runtime.embed_text_main_text_embed(pd.DataFrame([{"text": "document"}]), **kwargs)
        request.assert_called_once()
        assert request.call_args.kwargs["transform_config"].embedding_model == metadata["embedding_model_name"]
        assert request.call_args.kwargs["task_config"]["endpoint_url"] == _BatchEmbedCPUActor.DEFAULT_EMBED_INVOKE_URL
        driver_probe.assert_not_called()


def test_explicit_sink_identity_and_dimension_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    _, sink_kwargs = resolved_stages(
        tmp_path, EmbedParams(), vdb_kwargs={"embedding_model_name": "explicit/model", "vector_dim": 1024}
    )
    assert sink_kwargs["vdb_kwargs"]["embedding_model_name"] == "explicit/model"
    assert sink_kwargs["vdb_kwargs"]["vector_dim"] == 1024


@pytest.mark.parametrize(
    "requested_model,expected",
    [(None, NEMOTRON_35_EMBED_BF16_MODEL), (NEMOTRON_3_EMBED_MODEL, NEMOTRON_3_EMBED_BF16_MODEL)],
)
@pytest.mark.parametrize("driver_has_blackwell", [True, False])
def test_ray_default_does_not_use_driver_gpu_architecture(
    tmp_path, monkeypatch, requested_model, expected, driver_has_blackwell
):
    driver_probe = MagicMock(return_value=driver_has_blackwell)
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", driver_probe)
    # The same scheduler snapshot can represent Hopper, mixed, or unknown
    # workers, regardless of whether the driver itself has Blackwell GPUs.
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=2),
        available_resources=Resources(cpu_count=16, gpu_count=1),
    )
    original_params = EmbedParams(model_name=requested_model)
    actor_params, sink_kwargs = resolved_stages(tmp_path, original_params, resources=cluster)
    metadata = sink_kwargs["vdb_kwargs"]
    assert actor_params.model_name == actor_params.embed_model_name == metadata["embedding_model_name"] == expected
    assert actor_params.embed_model_revision == metadata["embedding_model_revision"] == get_hf_revision(expected)
    driver_probe.assert_not_called()
    assert original_params.model_name == requested_model
    from nemo_retriever.graph.retriever import Retriever

    query_kwargs = Retriever()._resolve_embed_kwargs(expected, None, metadata["embedding_model_revision"])
    assert query_kwargs["model_name"] == query_kwargs["embed_model_name"] == expected
    assert query_kwargs["embed_model_revision"] == actor_params.embed_model_revision
    driver_probe.assert_not_called()


@pytest.mark.parametrize(
    "checkpoint", [NEMOTRON_35_EMBED_BF16_MODEL, NEMOTRON_35_EMBED_NVFP4_MODEL, NEMOTRON_3_EMBED_BF16_MODEL]
)
def test_ray_preserves_explicit_checkpoint_precision(tmp_path, monkeypatch, checkpoint):
    driver_probe = MagicMock(side_effect=AssertionError("Driver CUDA must not select worker checkpoints"))
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", driver_probe)
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=8, gpu_count=1),
        available_resources=Resources(cpu_count=8, gpu_count=1),
    )
    actor_params, sink_kwargs = resolved_stages(tmp_path, EmbedParams(model_name=checkpoint), resources=cluster)
    assert actor_params.model_name == sink_kwargs["vdb_kwargs"]["embedding_model_name"] == checkpoint
    assert actor_params.embed_model_revision == sink_kwargs["vdb_kwargs"]["embedding_model_revision"]
    driver_probe.assert_not_called()
