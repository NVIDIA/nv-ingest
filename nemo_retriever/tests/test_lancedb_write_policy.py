# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

lancedb = pytest.importorskip("lancedb")

from nemo_retriever.common.vdb.lancedb import LanceDB


def _records(text: str = "hello", vector: list[float] | None = None) -> list[list[dict]]:
    return [
        [
            {
                "document_type": "text",
                "embedding": vector or [1.0, 0.0],
                "metadata": {
                    "content": text,
                    "content_metadata": {"page_number": 1},
                    "source_metadata": {"source_name": "doc.pdf"},
                },
            }
        ]
    ]


def _count_rows(uri: Path, table_name: str = "t") -> int:
    return int(lancedb.connect(str(uri)).open_table(table_name).count_rows())


def _image_only_records(vector: list[float] | None = None) -> list[list[dict]]:
    metadata = {
        "content": "",
        "content_metadata": {
            "page_number": 7,
            "type": "image",
        },
        "source_metadata": {"source_id": "scanned.pdf", "source_name": "scanned.pdf"},
    }
    record: dict = {"document_type": "image", "metadata": metadata}
    if vector is not None:
        record["embedding"] = vector
    return [[record]]


def _write_rows(tmp_path: Path, records: list[list[dict]], *, sparse: bool = False) -> list[dict]:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, sparse=sparse, create_index=False)
    op.run(records)
    return lancedb.connect(str(tmp_path)).open_table("t").to_arrow().to_pylist()


def test_overwrite_same_records_twice_keeps_row_count_stable(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)

    op.run(_records())
    assert _count_rows(tmp_path) == 1

    op.run(_records())
    assert _count_rows(tmp_path) == 1


def test_append_to_missing_table_creates_it(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())

    assert _count_rows(tmp_path) == 1


def test_append_same_records_twice_doubles_row_count(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    op.run(_records())
    with caplog.at_level(logging.WARNING):
        op.run(_records())

    assert _count_rows(tmp_path) == 2
    assert "Append mode does not deduplicate" in caplog.text


def test_append_with_matching_embedding_model_succeeds(tmp_path: Path) -> None:
    model_name = "nvidia/embedding-model-a"
    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name=model_name,
        create_index=False,
    ).run(_records())

    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name=model_name,
        overwrite=False,
        create_index=False,
    ).run(_records())

    assert _count_rows(tmp_path) == 2


def test_embedding_model_revision_is_recorded_and_readable(tmp_path: Path) -> None:
    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-a",
        embedding_model_revision="a" * 40,
        create_index=False,
    )

    op.run(_records())

    assert op.get_index_metadata("embedding_model_name") == "nvidia/embedding-model-a"
    assert op.get_index_metadata("embedding_model_revision") == "a" * 40


def test_vector_dimension_can_be_inferred_from_model_output(tmp_path: Path) -> None:
    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=None,
        embedding_model_name="nvidia/llama-embed-nemotron-8b",
        create_index=False,
    )

    op.run(_records(vector=[0.0] * 4096))

    table = lancedb.connect(str(tmp_path)).open_table("t")
    schema = table.schema() if callable(table.schema) else table.schema
    assert schema.field("vector").type.list_size == 4096


def test_append_with_inferred_dimension_uses_existing_table_schema(tmp_path: Path) -> None:
    kwargs = {
        "uri": str(tmp_path),
        "table_name": "t",
        "vector_dim": None,
        "embedding_model_name": "nvidia/llama-embed-nemotron-8b",
        "create_index": False,
    }
    LanceDB(**kwargs).run(_records(vector=[0.0] * 4096))

    LanceDB(**kwargs, overwrite=False).run(_records(vector=[0.0] * 4096))

    assert _count_rows(tmp_path) == 2


def test_append_with_mismatched_embedding_model_fails_before_write(tmp_path: Path) -> None:
    LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-a",
        create_index=False,
    ).run(_records())

    op = LanceDB(
        uri=str(tmp_path),
        table_name="t",
        vector_dim=2,
        embedding_model_name="nvidia/embedding-model-b",
        overwrite=False,
        create_index=False,
    )

    with pytest.raises(ValueError, match="cannot append vectors"):
        op.run(_records())

    assert _count_rows(tmp_path) == 1


@pytest.mark.parametrize(
    ("stored_revision", "incoming_revision", "error_pattern", "expected_rows"),
    [
        pytest.param("a" * 40, "b" * 40, "cannot append vectors from revision", 1, id="mismatch"),
        pytest.param("a" * 40, None, "without a known revision", 1, id="missing"),
        pytest.param("a" * 40, "a" * 40, None, 2, id="matching"),
        pytest.param(None, "a" * 40, None, 2, id="legacy-table"),
    ],
)
def test_append_revision_compatibility(
    tmp_path: Path,
    stored_revision: str | None,
    incoming_revision: str | None,
    error_pattern: str | None,
    expected_rows: int,
) -> None:
    common = {
        "uri": str(tmp_path),
        "table_name": "t",
        "vector_dim": 2,
        "embedding_model_name": "nvidia/embedding-model-a",
        "create_index": False,
    }
    LanceDB(**common, embedding_model_revision=stored_revision).run(_records())
    incoming = LanceDB(
        **common,
        embedding_model_revision=incoming_revision,
        overwrite=False,
    )

    if error_pattern is not None:
        with pytest.raises(ValueError, match=error_pattern):
            incoming.run(_records())
    else:
        incoming.run(_records())

    assert _count_rows(tmp_path) == expected_rows


def test_append_incompatible_schema_raises_clear_error(tmp_path: Path) -> None:
    LanceDB(uri=str(tmp_path), table_name="t", vector_dim=3, create_index=False).run(_records(vector=[1.0, 0.0, 0.0]))

    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, overwrite=False, create_index=False)

    with pytest.raises(ValueError, match="incompatible field 'vector'"):
        op.run(_records())


def test_create_index_kwarg_disables_index_build_without_shadowing_method(tmp_path: Path) -> None:
    op = LanceDB(uri=str(tmp_path), table_name="t", vector_dim=2, create_index=False)
    assert callable(op.create_index)
    assert op.build_index is False

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("write_to_index should not be called when create_index=False")

    op.write_to_index = fail_if_called  # type: ignore[method-assign]
    op.run(_records())

    assert _count_rows(tmp_path) == 1


@pytest.mark.parametrize(
    "caption",
    [pytest.param(None, id="empty"), pytest.param(" \n\t ", id="whitespace")],
)
def test_dense_write_stores_blank_canonical_image_record(tmp_path: Path, caption: str | None) -> None:
    records = _image_only_records([1.0, 0.0])
    if caption is not None:
        records[0][0]["metadata"]["image_metadata"] = {"caption": caption}

    table_rows = _write_rows(tmp_path, records)

    assert len(table_rows) == 1
    assert table_rows[0]["text"] == ""
    assert json.loads(table_rows[0]["metadata"]) == {"type": "image", "page_number": 7}
    assert json.loads(table_rows[0]["source"]) == {
        "source_id": "scanned.pdf",
        "source_name": "scanned.pdf",
    }


@pytest.mark.parametrize(
    "text",
    [pytest.param("", id="empty"), pytest.param(" \n\t ", id="whitespace")],
)
def test_dense_write_drops_blank_non_image_row(tmp_path: Path, text: str) -> None:
    table_rows = _write_rows(tmp_path, _records(text=text))

    assert table_rows == []


@pytest.mark.parametrize("missing_field", ["document_type", "content_metadata.type"])
def test_dense_write_requires_both_canonical_image_fields(tmp_path: Path, missing_field: str) -> None:
    records = _image_only_records([1.0, 0.0])
    if missing_field == "document_type":
        records[0][0]["document_type"] = "text"
    else:
        records[0][0]["metadata"]["content_metadata"].pop("type")

    table_rows = _write_rows(tmp_path, records)

    assert table_rows == []


@pytest.mark.parametrize(
    "vector",
    [pytest.param(None, id="missing"), pytest.param([1.0], id="wrong-length")],
)
def test_dense_write_drops_image_only_row_with_invalid_embedding(tmp_path: Path, vector: list[float] | None) -> None:
    table_rows = _write_rows(tmp_path, _image_only_records(vector))

    assert table_rows == []


def test_sparse_write_drops_image_only_row_without_text(tmp_path: Path) -> None:
    table_rows = _write_rows(tmp_path, _image_only_records([1.0, 0.0]), sparse=True)

    assert table_rows == []


def test_sparse_write_drops_whitespace_only_text(tmp_path: Path) -> None:
    table_rows = _write_rows(tmp_path, _records(text=" \n\t "), sparse=True)

    assert table_rows == []
