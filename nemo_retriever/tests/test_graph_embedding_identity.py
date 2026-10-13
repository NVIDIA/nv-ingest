# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph actor/index identities must agree after local/remote variant selection."""

from unittest.mock import MagicMock

import pytest

from nemo_retriever.common.params import EmbedParams, VdbUploadParams
from nemo_retriever.common.ray_resource_hueristics import Resources
from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph
from nemo_retriever.graph.operator_resolution import resolve_graph
from nemo_retriever.models import NEMOTRON_35_EMBED_BF16_MODEL, NEMOTRON_35_EMBED_NVFP4_MODEL
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.operators.vdb import IngestVdbOperator


def resolved_stages(tmp_path, params, *, gpu_count=1, vdb_kwargs=None):
    graph = build_post_extract_graph(
        embed_params=params,
        vdb_upload_params=VdbUploadParams(vdb_kwargs={"uri": str(tmp_path), **(vdb_kwargs or {})}),
        stage_order=("embed",),
        reshape_content_before_embed=False,
    )
    resolved = resolve_graph(graph, Resources(cpu_count=8, gpu_count=gpu_count))
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


def test_explicit_sink_identity_and_dimension_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    _, sink_kwargs = resolved_stages(
        tmp_path, EmbedParams(), vdb_kwargs={"embedding_model_name": "explicit/model", "vector_dim": 1024}
    )
    assert sink_kwargs["vdb_kwargs"]["embedding_model_name"] == "explicit/model"
    assert sink_kwargs["vdb_kwargs"]["vector_dim"] == 1024
