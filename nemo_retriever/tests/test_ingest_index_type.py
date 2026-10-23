# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed dense-index policy and real synthetic LanceDB construction contracts."""

from dataclasses import replace

import pytest

from nemo_retriever.ingest import plan as ingest


def request(tmp_path, **storage):
    document = tmp_path / "synthetic.txt"
    document.write_text("synthetic software fixture\n")
    return ingest.IngestPlanRequest(
        source=ingest.IngestSourceOptions(documents=[str(document)]),
        embed=ingest.IngestEmbedOptions(
            embed_model_name="nvidia/Nemotron-3.5-Embed-1B-BF16",
            local_ingest_embed_backend="hf",
        ),
        storage=ingest.IngestStorageOptions(lancedb_uri=str(tmp_path / "db"), **storage),
    )


@pytest.mark.parametrize("mode", ["auto", "dense", "hybrid", "sparse"])
def test_omitted_and_explicit_none_preserve_storage_plan(tmp_path, mode):
    original = request(tmp_path, index_mode=mode)
    omitted = ingest.resolve_ingest_plan(original)
    explicit = ingest.resolve_ingest_plan(replace(original, storage=replace(original.storage, index_type=None)))
    assert omitted.vdb_params == explicit.vdb_params
    assert "index_type" not in omitted.vdb_params.vdb_kwargs


@pytest.mark.parametrize("family", ["IVF_HNSW_SQ", "IVF_FLAT"])
def test_explicit_family_reaches_normal_vdb_operator_kwargs(tmp_path, family):
    plan = ingest.resolve_ingest_plan(request(tmp_path, index_mode="dense", index_type=family))
    assert plan.vdb_params.vdb_kwargs["index_type"] == family
    assert plan.vdb_params.to_ingest_operator_kwargs()["index_type"] == family


@pytest.mark.parametrize("invalid", ["", "ivf_flat", "IVF_PQ", "FLAT", 1, True, [], {}])
def test_invalid_family_rejected_before_document_expansion(tmp_path, monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid family must not touch documents or existing storage")

    monkeypatch.setattr(ingest, "expand_ingest_documents", forbidden)
    monkeypatch.setattr(ingest, "inspect_existing_lancedb_mode", forbidden)
    with pytest.raises(ValueError, match="storage.index_type must be"):
        ingest.resolve_ingest_plan(request(tmp_path, index_mode="dense", index_type=invalid))


@pytest.mark.parametrize("mode", ["auto", "hybrid", "sparse"])
def test_explicit_family_rejects_modes_without_dense_only_build(tmp_path, mode):
    with pytest.raises(ValueError, match="index_mode='dense'"):
        ingest.resolve_ingest_plan(request(tmp_path, index_mode=mode, index_type="IVF_FLAT"))


def test_explicit_family_rejects_append_before_opening_existing_table(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit family cannot inspect or change an append target")

    monkeypatch.setattr(ingest, "inspect_existing_lancedb_mode", forbidden)
    with pytest.raises(ValueError, match="requires overwrite=True"):
        ingest.resolve_ingest_plan(request(tmp_path, index_mode="dense", index_type="IVF_FLAT", overwrite=False))


@pytest.mark.parametrize("family", ["IVF_HNSW_SQ", "IVF_FLAT"])
def test_real_backend_builds_requested_family_from_typed_plan(tmp_path, family):
    import numpy as np

    from nemo_retriever.common.vdb.lancedb import LanceDB
    from nemo_retriever.common.vdb.lancedb_comparison import compare_dense_search

    # Synthetic vectors exercise the real backend; no checkpoint or model admission.
    vectors = np.random.default_rng(20261022).normal(size=(512, 8)).astype(np.float32).tolist()
    plan = ingest.resolve_ingest_plan(request(tmp_path, index_mode="dense", index_type=family))
    kwargs = dict(plan.vdb_params.vdb_kwargs)
    backend = LanceDB(**kwargs, expiration_cleanup_enabled=False)
    backend.run(
        [
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "embedding": vector,
                        "content": f"synthetic row {i}",
                        "content_metadata": {"page_number": i + 1, "type": "text"},
                        "source_metadata": {"source_id": "/synthetic/doc.txt", "source_name": "doc.txt"},
                    },
                }
                for i, vector in enumerate(vectors)
            ]
        ]
    )
    table = backend._connect().open_table(kwargs["table_name"])
    version = table.version
    indexed, exact, evidence = compare_dense_search(
        backend,
        [vectors[17]],
        top_k=10,
        expected_model_name=kwargs["embedding_model_name"],
        expected_model_revision=kwargs["embedding_model_revision"],
    )
    assert evidence["index_type"] == family
    assert evidence["table_rows"] == evidence["indexed_rows"] == 512
    assert evidence["unindexed_rows"] == 0 and evidence["vector_dimension"] == 8
    assert evidence["table_version"] == table.version == version
    assert indexed[0][0]["text"] == exact[0][0]["text"] == "synthetic row 17"
    if family == "IVF_FLAT":
        # The fixture's default probes cover its16 partitions. This assertion is
        # a small backend contract, not an exactness promise for other profiles.
        assert [hit["text"] for hit in indexed[0]] == [hit["text"] for hit in exact[0]]
