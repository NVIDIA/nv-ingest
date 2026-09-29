# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import lancedb
import pyarrow as pa
import pytest

from nemo_retriever.common.params import EmbedParams, VdbUploadParams
from nemo_retriever.ingest.index_mode import resolve_ingest_index_mode, resolve_vdb_upload_kwargs


@pytest.mark.parametrize(
    ("requested", "existing", "expected"),
    [
        ("auto", None, "hybrid"),
        ("dense", None, "dense"),
        ("hybrid", None, "hybrid"),
        ("sparse", None, "sparse"),
        ("auto", "dense", "dense"),
        ("dense", "dense", "dense"),
        ("hybrid", "dense", "hybrid"),
        ("auto", "hybrid", "hybrid"),
        ("hybrid", "hybrid", "hybrid"),
        ("auto", "sparse", "sparse"),
        ("sparse", "sparse", "sparse"),
    ],
)
def test_resolve_ingest_index_mode_compatible_transitions(requested, existing, expected) -> None:
    assert resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing) == expected


@pytest.mark.parametrize("requested", ["auto", "dense", "hybrid", "sparse"])
def test_resolve_ingest_index_mode_overwrite_ignores_existing_mode(requested) -> None:
    expected = "hybrid" if requested == "auto" else requested
    assert resolve_ingest_index_mode(requested, overwrite=True, existing_mode="sparse") == expected


@pytest.mark.parametrize(
    ("requested", "existing"),
    [
        ("dense", "hybrid"),
        ("sparse", "hybrid"),
        ("dense", "sparse"),
        ("hybrid", "sparse"),
        ("sparse", "dense"),
    ],
)
def test_resolve_ingest_index_mode_rejects_incompatible_append(requested, existing) -> None:
    with pytest.raises(ValueError, match="Cannot append"):
        resolve_ingest_index_mode(requested, overwrite=False, existing_mode=existing)


def _upload_kwargs(tmp_path, **vdb_kwargs):
    return resolve_vdb_upload_kwargs(VdbUploadParams(vdb_kwargs={"uri": str(tmp_path), **vdb_kwargs}))


@pytest.mark.parametrize(
    ("vdb_kwargs", "expected"),
    [
        ({}, {"hybrid": True}),
        ({"overwrite": False}, {"hybrid": True}),
        ({"sparse": False}, {"hybrid": True}),
        ({"hybrid": False}, {}),
        ({"sparse": True}, {}),
    ],
)
def test_resolve_vdb_upload_kwargs_new_tables_and_explicit_modes(tmp_path, vdb_kwargs, expected) -> None:
    assert _upload_kwargs(tmp_path, **vdb_kwargs) == {"uri": str(tmp_path), **vdb_kwargs, **expected}


@pytest.mark.parametrize(
    ("vector", "fts", "recorded_mode", "expected"),
    [
        (True, False, None, {"hybrid": False}),
        (True, True, None, {"hybrid": True}),
        (False, True, None, {"sparse": True}),
        (True, False, b"hybrid", {"hybrid": True}),
    ],
)
def test_resolve_vdb_upload_kwargs_append_keeps_the_table_mode(tmp_path, vector, fts, recorded_mode, expected) -> None:
    row = {"text": "alpha", **({"vector": [0.1, 0.2]} if vector else {})}
    fields = [pa.field("text", pa.string())] + ([pa.field("vector", pa.list_(pa.float32(), 2))] if vector else [])
    metadata = {b"nemo_retriever.retrieval_mode": recorded_mode} if recorded_mode else None
    table = lancedb.connect(str(tmp_path)).create_table("docs", data=[row], schema=pa.schema(fields, metadata=metadata))
    if fts:
        table.create_fts_index("text")

    resolved = _upload_kwargs(tmp_path, table_name="docs", overwrite=False)

    assert resolved == {"uri": str(tmp_path), "table_name": "docs", "overwrite": False, **expected}


LLAMA = "nvidia/llama-nemotron-embed-1b-v2"
REMOTE = "http://embed:8000/v1"


@pytest.mark.parametrize(
    ("embed_params", "vdb_kwargs", "expected"),
    [
        (EmbedParams(), {}, {"embedding_model_name": "nvidia/nemotron-3-embed-1b"}),
        (
            EmbedParams(model_name=LLAMA, embed_model_revision="abc123"),
            {},
            {"embedding_model_name": LLAMA, "embedding_model_revision": "abc123"},
        ),
        (EmbedParams(model_name=LLAMA, embed_invoke_url=REMOTE), {}, {"embedding_model_name": LLAMA}),
        # Remote requests send model_name, so an unset model_name means the default model.
        (
            EmbedParams(embed_model_name=LLAMA, embed_invoke_url=REMOTE),
            {},
            {"embedding_model_name": "nvidia/nemotron-3-embed-1b"},
        ),
        # Locally, a GPU actor would use embed_model_name and a CPU fallback model_name.
        (EmbedParams(embed_model_name=LLAMA), {}, {}),
        (EmbedParams(), {"embedding_model_name": "custom/model"}, {"embedding_model_name": "custom/model"}),
        (EmbedParams(), {"sparse": True}, {}),
    ],
)
def test_resolve_vdb_upload_kwargs_records_the_embed_model(tmp_path, embed_params, vdb_kwargs, expected) -> None:
    params = VdbUploadParams(vdb_kwargs={"uri": str(tmp_path), **vdb_kwargs})

    resolved = resolve_vdb_upload_kwargs(params, embed_params)

    assert {key: value for key, value in resolved.items() if key.startswith("embedding_model")} == expected
