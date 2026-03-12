# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.ingest_modes.lancedb_utils."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Stub heavy internal modules so ``nemo_retriever.ingest_modes`` can be
# imported in lightweight CI (no ray, torch, nemotron_*, pypdfium2).
#
# IMPORTANT: We do NOT stub ``nemo_retriever.ingest_modes.inprocess`` itself
# because test_multimodal_embed.py needs the real module.  Instead we stub the
# same transitive-heavy deps that test_multimodal_embed.py does.
_HEAVY_INTERNAL = [
    "nemo_retriever.ingest_modes.batch",
    "nemo_retriever.ingest_modes.fused",
    "nemo_retriever.ingest_modes.online",
    "nemo_retriever.model.local",
    "nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder",
    "nemo_retriever.model.local.nemotron_page_elements_v3",
    "nemo_retriever.model.local.nemotron_ocr_v1",
    "nemo_retriever.model.local.nemotron_table_structure_v1",
    "nemo_retriever.model.local.nemotron_graphic_elements_v1",
    "nemo_retriever.page_elements",
    "nemo_retriever.page_elements.page_elements",
    "nemo_retriever.ocr",
    "nemo_retriever.ocr.ocr",
    "nemo_retriever.pdf",
    "nemo_retriever.pdf.__main__",
    "nemo_retriever.pdf.config",
    "nemo_retriever.pdf.io",
    "nemo_retriever.pdf.stage",
    "nemo_retriever.pdf.extract",
    "nemo_retriever.pdf.split",
    "nemo_retriever.chart",
    "nemo_retriever.chart.chart_detection",
    "nemo_retriever.chart.commands",
    "nemo_retriever.chart.config",
    "nemo_retriever.chart.processor",
]
for _mod_name in _HEAVY_INTERNAL:
    sys.modules.setdefault(_mod_name, MagicMock())

from nemo_retriever.ingest_modes.lancedb_utils import (  # noqa: E402
    build_lancedb_row,
    build_lancedb_rows,
    create_or_append_lancedb_table,
    extract_embedding_from_row,
    extract_source_path_and_page,
    infer_vector_dim,
    lancedb_schema,
)


class TestExtractEmbeddingFromRow:
    def test_from_metadata(self):
        row = SimpleNamespace(metadata={"embedding": [1.0, 2.0, 3.0]})
        assert extract_embedding_from_row(row) == [1.0, 2.0, 3.0]

    def test_from_embedding_column(self):
        row = SimpleNamespace(
            metadata=None,
            text_embeddings_1b_v2={"embedding": [4.0, 5.0]},
        )
        assert extract_embedding_from_row(row) == [4.0, 5.0]

    def test_custom_column(self):
        row = SimpleNamespace(metadata=None, my_col={"vec": [6.0]})
        assert extract_embedding_from_row(row, embedding_column="my_col", embedding_key="vec") == [6.0]

    def test_returns_none_when_missing(self):
        row = SimpleNamespace(metadata=None)
        assert extract_embedding_from_row(row) is None

    def test_empty_embedding_returns_none(self):
        row = SimpleNamespace(metadata={"embedding": []})
        assert extract_embedding_from_row(row) is None


class TestExtractSourcePathAndPage:
    def test_from_direct_attrs(self):
        row = SimpleNamespace(path="/docs/file.pdf", page_number=3, metadata=None)
        assert extract_source_path_and_page(row) == ("/docs/file.pdf", 3)

    def test_from_metadata_source_path(self):
        row = SimpleNamespace(path="", page_number=None, metadata={"source_path": "/meta/path.pdf"})
        assert extract_source_path_and_page(row) == ("/meta/path.pdf", -1)

    def test_from_content_metadata_hierarchy(self):
        row = SimpleNamespace(
            path="",
            page_number=None,
            metadata={"content_metadata": {"hierarchy": {"page": 7}}},
        )
        path, page = extract_source_path_and_page(row)
        assert page == 7

    def test_defaults_when_missing(self):
        row = SimpleNamespace()
        assert extract_source_path_and_page(row) == ("", -1)


class TestBuildLancedbRow:
    def _row(self, **kwargs):
        defaults = {
            "metadata": {"embedding": [0.1, 0.2]},
            "path": "/docs/test.pdf",
            "page_number": 1,
            "text": "hello world",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_returns_dict_with_expected_keys(self):
        result = build_lancedb_row(self._row())
        assert result is not None
        assert set(result.keys()) == {
            "vector",
            "pdf_page",
            "filename",
            "pdf_basename",
            "page_number",
            "source_id",
            "path",
            "metadata",
            "source",
            "text",
        }

    def test_vector_extracted(self):
        result = build_lancedb_row(self._row())
        assert result["vector"] == [0.1, 0.2]

    def test_path_fields(self):
        result = build_lancedb_row(self._row())
        assert result["filename"] == "test.pdf"
        assert result["pdf_basename"] == "test"
        assert result["pdf_page"] == "test_1"

    def test_text_included(self):
        result = build_lancedb_row(self._row())
        assert result["text"] == "hello world"

    def test_text_excluded(self):
        result = build_lancedb_row(self._row(), include_text=False)
        assert result["text"] == ""

    def test_metadata_json(self):
        result = build_lancedb_row(self._row())
        meta = json.loads(result["metadata"])
        assert meta["page_number"] == 1
        assert meta["pdf_page"] == "test_1"

    def test_returns_none_when_no_embedding(self):
        row = SimpleNamespace(metadata=None, path="/x.pdf", page_number=1, text="hi")
        assert build_lancedb_row(row) is None

    def test_detection_metadata_included(self):
        row = self._row(
            page_elements_v3_num_detections=5,
            page_elements_v3_counts_by_label={"text": 3, "figure": 2},
            table=[{}, {}],
        )
        result = build_lancedb_row(row)
        meta = json.loads(result["metadata"])
        assert meta["page_elements_v3_num_detections"] == 5
        assert meta["page_elements_v3_counts_by_label"] == {"text": 3, "figure": 2}
        assert meta["ocr_table_detections"] == 2


class TestBuildLancedbRows:
    def test_filters_rows_without_embeddings(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"metadata": {"embedding": [1.0]}, "path": "/a.pdf", "page_number": 1, "text": "a"},
                {"metadata": {}, "path": "/b.pdf", "page_number": 1, "text": "b"},
            ]
        )
        rows = build_lancedb_rows(df)
        assert len(rows) == 1
        assert rows[0]["vector"] == [1.0]


class TestLancedbSchema:
    def test_returns_schema_with_correct_fields(self):
        pytest.importorskip("pyarrow")
        schema = lancedb_schema(768)
        names = [f.name for f in schema]
        assert "vector" in names
        assert "text" in names
        assert "metadata" in names
        assert "source" in names
        assert "source_id" in names
        assert len(names) == 10


class TestInferVectorDim:
    def test_returns_dim(self):
        assert infer_vector_dim([{"vector": [1.0, 2.0, 3.0]}]) == 3

    def test_returns_zero_when_empty(self):
        assert infer_vector_dim([]) == 0
        assert infer_vector_dim([{"vector": []}]) == 0


class TestCreateOrAppendLancedbTable:
    def test_overwrite_calls_create(self):
        from unittest.mock import MagicMock

        db = MagicMock()
        schema = MagicMock()
        rows = [{"a": 1}]
        create_or_append_lancedb_table(db, "test", rows, schema, overwrite=True)
        db.create_table.assert_called_once_with("test", data=[{"a": 1}], schema=schema, mode="overwrite")

    def test_append_opens_then_adds(self):
        from unittest.mock import MagicMock

        db = MagicMock()
        table = MagicMock()
        db.open_table.return_value = table
        rows = [{"a": 1}]
        result = create_or_append_lancedb_table(db, "t", rows, MagicMock(), overwrite=False)
        db.open_table.assert_called_once_with("t")
        table.add.assert_called_once()
        assert result is table

    def test_append_falls_back_to_create(self):
        from unittest.mock import MagicMock

        db = MagicMock()
        db.open_table.side_effect = Exception("not found")
        schema = MagicMock()
        rows = [{"a": 1}]
        create_or_append_lancedb_table(db, "t", rows, schema, overwrite=False)
        db.create_table.assert_called_once_with("t", data=[{"a": 1}], schema=schema, mode="create")
