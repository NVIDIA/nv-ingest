# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    UnsupportedVDBOperation,
    VDB,
)
from nemo_retriever.common.vdb.records import RetrievalContractError, VdbUploadError
from nemo_retriever.operators.vdb import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.operators import vdb as vdb_operator_module
from nemo_retriever.operators.vdb import PutVdbOperator


class _CollectionContractStub(VDB):
    """Implement required collection methods that these operator tests do not exercise."""

    def _unexpected_collection_operation(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Unexpected collection operation")

    create_collection = _unexpected_collection_operation
    get_collection = _unexpected_collection_operation
    list_collections = _unexpected_collection_operation
    update_collection = _unexpected_collection_operation
    delete_collection = _unexpected_collection_operation
    get_document = _unexpected_collection_operation
    list_documents = _unexpected_collection_operation
    delete_document = _unexpected_collection_operation
    write_collection = _unexpected_collection_operation
    retrieve_collection = _unexpected_collection_operation


class FakeVDB(_CollectionContractStub):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[Any] = []
        self.retrieval_calls: list[tuple[Any, dict[str, Any]]] = []
        self.put_calls: list[tuple[Any, dict[str, Any]]] = []
        self.write_collection_calls: list[tuple[Any, CollectionWriteContext]] = []
        self.retrieve_collection_calls: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, vectors: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.retrieval_calls.append((vectors, kwargs))
        return [
            [
                {
                    "_distance": 0.12,
                    "bbox_xyxy_norm": "[0, 0, 1, 1]",
                    "content_type": "table",
                    "entity": {
                        "text": "retrieved chunk",
                        "source": {"source_id": "doc-a.pdf"},
                        "content_metadata": {"page_number": 1},
                    },
                    "stored_image_uri": "file:///tmp/page.png",
                }
            ]
        ]

    def run(self, records: Any) -> dict[str, Any]:
        self.run_calls.append(records)
        return {"records": records}

    def put(self, records: list, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append((records, dict(kwargs)))
        return {"put": sum(len(b) for b in records)}

    def write_collection(self, records: list, *, context: CollectionWriteContext) -> CollectionWriteResult:
        self.write_collection_calls.append((records, context))
        return CollectionWriteResult(written=sum(len(batch) for batch in records), total_rows=7)

    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        call_kwargs = {
            "scope": scope,
            "collection_name": collection_name,
            "query_texts": query_texts,
            "top_k": top_k,
            **kwargs,
        }
        self.retrieve_collection_calls.append((vectors, call_kwargs))
        return [
            [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "text": "retrieved chunk",
                    "distance": 0.12,
                    "filename": "doc-a.pdf",
                    "page_number": 1,
                    "content_type": "table",
                    "source": "doc-a.pdf",
                    "source_id": "doc-a.pdf",
                    "stored_image_uri": "file:///tmp/page.png",
                    "bbox": [0, 0, 1, 1],
                    "metadata": {},
                    "physical_table": "private-table",
                    "lancedb_uri": "/private/vector-store",
                }
            ]
        ], ["dense"]


class InvalidCollectionVDB(FakeVDB):
    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result

    def retrieve_collection(self, vectors: list, **kwargs: Any) -> Any:
        return self.result


def _graph_rows() -> list[dict[str, Any]]:
    return [
        {
            "text": "first chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "path": "/tmp/doc-a.pdf",
            "page_number": 1,
            "metadata": {"content_metadata": {"type": "text"}},
        },
        {
            "text": "second chunk",
            "text_embeddings_1b_v2": {"embedding": [0.2] * 2048},
            "path": "/tmp/doc-a.pdf",
            "page_number": 2,
            "metadata": {"content_metadata": {"type": "text"}},
        },
    ]


def test_process_returns_original_graph_rows_and_delegates_converted_records_to_run() -> None:
    data = _graph_rows()
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)

    assert operator.preprocess(data) is data
    assert operator.process(data) is data
    assert operator.postprocess(data) is data
    assert vdb.run_calls[0][0][0]["document_type"] == "text"
    assert vdb.run_calls[0][0][0]["metadata"]["content"] == "first chunk"
    assert vdb.run_calls[0][0][0]["metadata"]["embedding"] == [0.1] * 2048


def test_vdb_op_constructs_client_vdb(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_kwargs: dict[str, Any] = {}

    class ConstructedFakeVDB(FakeVDB):
        def __init__(self, **kwargs: Any) -> None:
            constructed_kwargs.update(kwargs)
            super().__init__(**kwargs)

    def fake_get_vdb_op_cls(vdb_op: str) -> type[ConstructedFakeVDB]:
        assert vdb_op == "fake"
        return ConstructedFakeVDB

    monkeypatch.setattr(vdb_operator_module, "get_vdb_op_cls", fake_get_vdb_op_cls)

    operator = IngestVdbOperator(vdb_op="fake", vdb_kwargs={"answer": 42})

    assert constructed_kwargs == {"answer": 42}
    assert operator.process(_graph_rows()) is not None


def test_ingest_operator_converts_graph_rows_to_client_vdb_records() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator(data) is data

    assert vdb.run_calls == [
        [
            [
                {
                    "document_type": "text",
                    "metadata": {
                        "embedding": [0.1] * 2048,
                        "content": "graph chunk",
                        "content_metadata": {"page_number": 7},
                        "source_metadata": {
                            "source_id": "/tmp/doc-a.pdf",
                            "source_name": "doc-a.pdf",
                        },
                    },
                }
            ]
        ]
    ]


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"text": "extracted chunk"}, id="text"),
        pytest.param({"text": "", "_image_b64": "page-image"}, id="image"),
    ],
)
def test_ingest_operator_rejects_uploadable_dataframe_without_embeddings(row: dict[str, Any]) -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = pd.DataFrame([row])

    with pytest.raises(
        ValueError,
        match="vdb_upload requires embedded records, but no embeddings were found",
    ):
        operator.process(data)
    assert vdb.run_calls == []


def test_ingest_operator_rejects_unembedded_content_when_embedded_row_is_filtered() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = pd.DataFrame(
        [
            {"text": "", "metadata": {"embedding": [0.1]}},
            {"text": "extracted chunk"},
        ]
    )

    with pytest.raises(
        ValueError,
        match="vdb_upload requires embedded records, but no embeddings were found",
    ):
        operator.process(data)
    assert vdb.run_calls == []


@pytest.mark.parametrize(
    "text",
    [pytest.param("", id="empty"), pytest.param(" \n\t ", id="whitespace")],
)
def test_ingest_operator_retains_embedded_blank_image_row_without_text_fidelity(
    text: str,
) -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": text,
            "document_type": "structured",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "_image_b64": "page-image",
            "source_id": "/tmp/scanned.pdf",
            "page_number": 7,
            "metadata": {"content_metadata": {"type": "text", "fidelity": "verbatim"}},
        }
    ]

    assert operator(data) is data

    record = vdb.run_calls[0][0][0]
    assert record["document_type"] == "image"
    assert record["metadata"]["content"] == ""
    assert record["metadata"]["embedding"] == [0.1] * 2048
    assert record["metadata"]["content_metadata"] == {
        "type": "image",
        "page_number": 7,
    }
    assert record["metadata"]["source_metadata"] == {
        "source_id": "/tmp/scanned.pdf",
        "source_name": "scanned.pdf",
    }


@pytest.mark.parametrize(
    "image_payload",
    [
        pytest.param(b"page-image", id="bytes"),
        pytest.param(np.ones((2, 2), dtype=np.uint8), id="numpy"),
    ],
)
def test_ingest_operator_rejects_noncanonical_image_payload_without_truthiness(
    image_payload: Any,
) -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": "",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "_image_b64": image_payload,
            "source_id": "/tmp/scanned.pdf",
            "page_number": 7,
        }
    ]

    with pytest.raises(
        VdbUploadError,
        match=r"none were uploadable; .*missing searchable text or image backing=1",
    ):
        operator(data)
    assert vdb.run_calls == []


@pytest.mark.parametrize("uri_field", ["_stored_image_uri", "stored_image_uri"])
def test_ingest_operator_retains_image_only_row_with_stored_image_uri(
    uri_field: str,
) -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": "",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            uri_field: "file:///tmp/scanned-page-7.png",
            "source_id": "/tmp/scanned.pdf",
            "page_number": 7,
        }
    ]

    assert operator(data) is data
    record = vdb.run_calls[0][0][0]
    assert record["document_type"] == "image"
    assert record["metadata"]["content_metadata"] == {
        "type": "image",
        "page_number": 7,
        "stored_image_uri": "file:///tmp/scanned-page-7.png",
        "uploaded_image_uri": "file:///tmp/scanned-page-7.png",
    }


def test_ingest_operator_rejects_nonempty_batch_with_zero_uploadable_records() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": "",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/empty.pdf",
            "page_number": 1,
        }
    ]

    with pytest.raises(
        VdbUploadError,
        match=r"received 1 row\(s\), but none were uploadable; .*missing searchable text or image backing=1",
    ):
        operator(data)
    assert vdb.run_calls == []


def test_ingest_operator_rejects_runtime_empty_vectors_before_backend_call() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = [
        {
            "text": text,
            "text_embeddings_1b_v2": {
                "embedding": [],
                "error": "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause",
            },
        }
        for text in ("valid neighbor before", "valid neighbor after")
    ]

    with pytest.raises(VdbUploadError, match=r"none were uploadable"):
        operator(data)

    assert vdb.run_calls == []


def test_ingest_operator_reports_upstream_error_counts_without_payloads() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = pd.DataFrame(
        [
            {
                "text": "",
                "metadata": {
                    "source_path": "/tmp/scanned.pdf",
                    "error": {
                        "stage": "ocr-nvapi-secret",
                        "type": "CustomerDocumentText",
                        "message": "private document content Authorization: Bearer token-secret",
                    },
                    "exception": "password=credential-secret",
                    "traceback": "Traceback containing private customer text",
                },
            }
        ]
    )

    with pytest.raises(
        VdbUploadError,
        match=r"reported 3 structured row error\(s\) \(error=1, exception=1, traceback=1\)",
    ) as exc_info:
        operator.process(data)
    rendered = str(exc_info.value)
    assert "payloads are omitted because they may contain sensitive data" in rendered
    assert "nvapi-secret" not in rendered
    assert "CustomerDocumentText" not in rendered
    assert "private document content" not in rendered
    assert "token-secret" not in rendered
    assert "credential-secret" not in rendered
    assert "private customer text" not in rendered
    assert vdb.run_calls == []


def test_ingest_operator_allows_genuinely_empty_dataframe() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    data = pd.DataFrame()

    assert operator.process(data) is data
    assert vdb.run_calls == []


def test_retrieve_operator_delegates_vectors_to_retrieval() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(vdb=vdb, vdb_kwargs={"collection_name": "docs", "model_name": "embedder"})

    result = operator.process([[0.1, 0.2]], top_k=3)

    assert result == [
        [
            {
                "text": "retrieved chunk",
                "metadata": {"page_number": 1},
                "source": "doc-a.pdf",
                "source_id": "doc-a.pdf",
                "path": "doc-a.pdf",
                "page_number": 1,
                "pdf_basename": "doc-a",
                "pdf_page": "doc-a_1",
                "_distance": 0.12,
                "stored_image_uri": "file:///tmp/page.png",
                "content_type": "table",
                "bbox_xyxy_norm": "[0, 0, 1, 1]",
            }
        ]
    ]
    assert vdb.retrieval_calls == [
        (
            [[0.1, 0.2]],
            {"collection_name": "docs", "model_name": "embedder", "top_k": 3},
        )
    ]


def test_retrieve_operator_reads_index_metadata_from_any_vdb() -> None:
    class MetadataVDB(FakeVDB):
        def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
            assert kwargs == {"collection_name": "docs"}
            return {"embedding_model_name": "acme/embed", "retrieval_mode": "dense"}.get(key)

    operator = RetrieveVdbOperator(vdb=MetadataVDB(), vdb_kwargs={"collection_name": "docs"})

    assert operator.get_index_metadata("embedding_model_name") == "acme/embed"
    assert operator.get_index_metadata("retrieval_mode") == "dense"


def test_retrieve_operator_forwards_runtime_query_texts() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(
        vdb=vdb,
        vdb_kwargs={
            "collection_name": "docs",
            "model_name": "embedder",
            "hybrid": True,
            "query_texts": ["stale"],
        },
    )

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [
        (
            [[0.1, 0.2]],
            {
                "collection_name": "docs",
                "model_name": "embedder",
                "hybrid": True,
                "top_k": 3,
                "query_texts": ["current"],
            },
        )
    ]


def test_retrieve_operator_forwards_query_texts_for_hybrid_vdb_instance() -> None:
    vdb = FakeVDB(hybrid=True)
    operator = RetrieveVdbOperator(vdb=vdb)

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"top_k": 3, "query_texts": ["current"]})]


def test_retrieve_operator_respects_dense_override_for_hybrid_vdb_instance() -> None:
    vdb = FakeVDB(hybrid=True)
    operator = RetrieveVdbOperator(vdb=vdb)

    operator.process([[0.1, 0.2]], top_k=3, hybrid=False, query_texts=["current"])

    assert vdb.retrieval_calls == [([[0.1, 0.2]], {"top_k": 3, "hybrid": False})]


def test_retrieve_operator_does_not_forward_query_texts_for_dense_retrieval() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(vdb=vdb, vdb_kwargs={"collection_name": "docs", "model_name": "embedder"})

    operator.process([[0.1, 0.2]], top_k=3, query_texts=["current"])

    assert vdb.retrieval_calls == [
        (
            [[0.1, 0.2]],
            {"collection_name": "docs", "model_name": "embedder", "top_k": 3},
        )
    ]


def test_constructor_requires_exactly_one_vdb_source() -> None:
    with pytest.raises(ValueError, match="Either vdb or vdb_op is required"):
        IngestVdbOperator()

    with pytest.raises(ValueError, match="Pass either vdb or vdb_op"):
        IngestVdbOperator(vdb=FakeVDB(), vdb_op="lancedb")


# ──────────────────────────────────────────────────────────────────────────────
# PutVdbOperator
# ──────────────────────────────────────────────────────────────────────────────


class _StubPutVDB(_CollectionContractStub):
    """VDB subclass that intentionally does NOT override ``put``.

    Used to exercise the construction-time guard in
    :class:`PutVdbOperator.__init__`, which compares
    ``type(self._vdb).put is VDB.put`` to detect backends that
    inherit the base-class ``NotImplementedError`` stub.

    Note: this class being instantiable at all is itself a regression
    check — :meth:`VDB.put` must NOT be decorated with
    ``@abstractmethod``; otherwise ABC machinery would reject this class
    before the operator-level guard could run, making the guard dead code.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, queries: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return []

    def run(self, records: Any) -> None:
        return None


def test_put_operator_rejects_vdb_without_put_override() -> None:
    """Backends inheriting the ``VDB.put`` stub fail fast at construction."""
    stub = _StubPutVDB()
    # Sanity-check the precondition the guard relies on: the subclass really
    # is using the inherited stub, not its own implementation. If this ever
    # fails, the guard's identity comparison would silently never fire.
    assert type(stub).put is VDB.put

    with pytest.raises(NotImplementedError, match=r"does not implement put"):
        PutVdbOperator(vdb=stub)


def test_put_operator_delegates_records_with_configured_key_and_table_name() -> None:
    """Happy path: nv-ingest-converted records reach ``vdb.put`` with the configured key/table."""
    vdb = FakeVDB()
    operator = PutVdbOperator(vdb=vdb, key="entity_id", table_name="entities")

    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator(data) is data

    assert vdb.run_calls == []
    assert len(vdb.put_calls) == 1
    call_records, call_kwargs = vdb.put_calls[0]
    assert call_kwargs == {"table_name": "entities", "key": "entity_id"}
    # The records that reach the backend must already be in nv-ingest-client
    # shape (same conversion IngestVdbOperator performs), not the flat graph rows.
    assert call_records == [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1] * 2048,
                    "content": "graph chunk",
                    "content_metadata": {"page_number": 7},
                    "source_metadata": {
                        "source_id": "/tmp/doc-a.pdf",
                        "source_name": "doc-a.pdf",
                    },
                },
            }
        ]
    ]


def test_custom_vdb_stream_capability_and_legacy_fallback() -> None:
    pulls: list[int] = []

    class LegacyDuckVDB:
        def run(self, records) -> None:
            pass

    class StreamingFakeVDB(FakeVDB):
        supports_stream_ingest = True

        def __init__(self) -> None:
            super().__init__()
            self.stream_records: list[dict[str, Any]] = []

        def stream_ingest(self, records) -> None:
            assert pulls == []
            assert not isinstance(records, (list, tuple, pd.DataFrame))
            self.stream_records = list(records)

    class EarlyReturningVDB(StreamingFakeVDB):
        def stream_ingest(self, records) -> None:
            next(iter(records))

    def batches():
        pulls.append(0)
        first = pd.DataFrame([_graph_rows()[0]])
        first["document_type"] = pd.Series([None], dtype="string[pyarrow]")
        yield first
        pulls.append(1)
        yield pd.DataFrame([_graph_rows()[1]])

    streaming_vdb = StreamingFakeVDB()
    streaming = IngestVdbOperator(vdb=streaming_vdb)
    legacy_vdb = FakeVDB()
    legacy = IngestVdbOperator(vdb=legacy_vdb)
    legacy_duck = IngestVdbOperator(vdb=LegacyDuckVDB())
    put = PutVdbOperator(vdb=streaming_vdb)

    assert streaming._stream_ingest(batches()) == 2
    assert pulls == [0, 1]
    assert streaming_vdb.run_calls == []
    assert [record["document_type"] for record in streaming_vdb.stream_records] == ["text", "text"]
    assert [record["metadata"]["content"] for record in streaming_vdb.stream_records] == [
        "first chunk",
        "second chunk",
    ]

    partial_rows = _graph_rows()
    partial_rows[1]["text_embeddings_1b_v2"] = {"embedding": []}
    fail_closed_vdb = StreamingFakeVDB()
    pulls.clear()
    with pytest.raises(VdbUploadError, match="refusing a partial write"):
        IngestVdbOperator(vdb=fail_closed_vdb)._stream_ingest([pd.DataFrame(partial_rows)])
    assert fail_closed_vdb.stream_records == []

    legacy.process(pd.DataFrame(_graph_rows()))
    assert len(legacy_vdb.run_calls) == 1
    with pytest.raises(UnsupportedVDBOperation, match="does not implement stream_ingest"):
        legacy._stream_ingest(batches())
    with pytest.raises(UnsupportedVDBOperation, match="does not implement stream_ingest"):
        legacy_duck._stream_ingest(batches())
    with pytest.raises(UnsupportedVDBOperation, match="does not implement stream_ingest"):
        put._stream_ingest(batches())
    with pytest.raises(RuntimeError, match="returned before consuming the record stream"):
        operator = IngestVdbOperator(vdb=EarlyReturningVDB())
        operator._stream_ingest([pd.DataFrame(_graph_rows())])


def test_put_operator_merges_sidecar_metadata_into_records_before_put() -> None:
    """Sidecar kwargs are split out from ``vdb_kwargs`` and applied before delegation."""
    vdb = FakeVDB()
    meta_df = pd.DataFrame(
        {
            "source_id": ["/tmp/doc-a.pdf"],
            "category": ["legal"],
        }
    )
    operator = PutVdbOperator(
        vdb=vdb,
        vdb_kwargs={
            "meta_dataframe": meta_df,
            "meta_source_field": "source_id",
            "meta_fields": ["category"],
            "meta_join_key": "source_id",
        },
        key="id",
        table_name="my_table",
    )

    data = [
        {
            "text": "graph chunk",
            "text_embeddings_1b_v2": {"embedding": [0.1] * 2048},
            "source_id": "/tmp/doc-a.pdf",
            "page_number": 7,
        }
    ]

    assert operator.process(data) is data

    assert len(vdb.put_calls) == 1
    call_records, call_kwargs = vdb.put_calls[0]
    assert call_kwargs == {"table_name": "my_table", "key": "id"}
    merged_content_meta = call_records[0][0]["metadata"]["content_metadata"]
    # Sidecar column merged in alongside the per-row ``page_number``.
    assert merged_content_meta["category"] == "legal"
    assert merged_content_meta["page_number"] == 7


def test_ingest_operator_preserves_sidecar_kwargs_for_graph_reconstruction() -> None:
    meta_df = pd.DataFrame({"source_id": ["/tmp/doc-a.pdf"], "category": ["legal"]})
    operator = IngestVdbOperator(
        vdb=FakeVDB(),
        vdb_kwargs={
            "meta_dataframe": meta_df,
            "meta_source_field": "source_id",
            "meta_fields": ["category"],
        },
    )

    reconstructed_kwargs = operator.get_constructor_kwargs()["vdb_kwargs"]

    assert reconstructed_kwargs["meta_dataframe"] is meta_df
    assert reconstructed_kwargs["meta_source_field"] == "source_id"
    assert reconstructed_kwargs["meta_fields"] == ["category"]


def test_ingest_operator_preserves_canonical_batches_for_collection_write() -> None:
    vdb = FakeVDB()
    operator = IngestVdbOperator(vdb=vdb)
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "embedding": [0.1, 0.2],
                    "content": "canonical chunk",
                    "content_metadata": {"page_number": 9},
                    "source_metadata": {"source_id": "/tmp/canonical.pdf"},
                },
            }
        ]
    ]
    context = CollectionWriteContext(
        scope="tenant-a",
        collection_name="papers",
        document_id="document-1",
        document_version="version-1",
        content_sha256="a" * 64,
        filename="canonical.pdf",
        job_id="job-1",
        operation="replace",
    )

    result = operator.process(records, collection_context=context)

    assert result == CollectionWriteResult(written=1, total_rows=7)
    assert vdb.write_collection_calls == [(records, context)]
    assert vdb.write_collection_calls[0][0] is records
    assert vdb.run_calls == []


def test_ingest_operator_rejects_empty_collection_write() -> None:
    operator = IngestVdbOperator(vdb=FakeVDB())
    context = CollectionWriteContext(
        scope="tenant-a",
        collection_name="papers",
        document_id="document-1",
        document_version="version-1",
        content_sha256="a" * 64,
        filename="empty.pdf",
    )

    with pytest.raises(ValueError, match="at least one canonical VDB record"):
        operator.process([], collection_context=context)


def test_retrieve_operator_dispatches_explicit_collection_context() -> None:
    vdb = FakeVDB()
    operator = RetrieveVdbOperator(vdb=vdb, vdb_kwargs={"model_name": "embedder"})

    hits, strategies = operator.process(
        [[0.1, 0.2]],
        scope="tenant-a",
        collection_name="papers",
        top_k=3,
        query_texts=["current"],
    )

    assert strategies == ["dense"]
    assert hits[0][0]["text"] == "retrieved chunk"
    assert "physical_table" not in hits[0][0]
    assert "lancedb_uri" not in hits[0][0]
    assert vdb.retrieve_collection_calls == [
        (
            [[0.1, 0.2]],
            {
                "scope": "tenant-a",
                "collection_name": "papers",
                "query_texts": ["current"],
                "top_k": 3,
                "model_name": "embedder",
            },
        )
    ]


@pytest.mark.parametrize(
    "result",
    [
        None,
        ([], ["dense"]),
        ([[42]], ["dense"]),
        ([[{"text": "missing collection identity"}]], ["dense"]),
        ([[{"chunk_id": "chunk-1"}]], []),
    ],
)
def test_retrieve_operator_rejects_malformed_collection_results(result: Any) -> None:
    operator = RetrieveVdbOperator(vdb=InvalidCollectionVDB(result))

    with pytest.raises(RetrievalContractError):
        operator.process(
            [[0.1, 0.2]],
            scope="tenant-a",
            collection_name="papers",
            query_texts=["query"],
        )


def test_retrieve_operator_rejects_partial_collection_context() -> None:
    operator = RetrieveVdbOperator(vdb=FakeVDB())

    with pytest.raises(ValueError, match="both scope and collection_name"):
        operator.process([[0.1, 0.2]], scope="tenant-a")
