import ast
import json
from types import SimpleNamespace

import numpy as np

from nemo_retriever.common.vdb.lancedb_bulk import _build_lancedb_rows_from_df
from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row


def test_build_lancedb_row_persists_normalized_content_type() -> None:
    row = SimpleNamespace(
        path="/tmp/doc_a.pdf",
        page_number=7,
        embedding=np.array([0.5, 0.25], dtype=np.float32),
        metadata={"source_path": "/tmp/doc_a.pdf"},
        text="table text",
        _content_type="table_caption",
    )

    row_out = build_lancedb_row(row, embedding=row.embedding)

    assert row_out is not None
    assert row_out["vector"] == [0.5, 0.25]
    metadata = json.loads(row_out["metadata"])
    assert metadata["_content_type"] == "table"


def test_build_lancedb_rows_from_df_persists_normalized_content_type() -> None:
    rows = [
        {
            "path": "/tmp/doc_b.pdf",
            "page_number": 3,
            "text": "chart text",
            "_content_type": "chart_caption",
            "embedding": [0.3, 0.4],
            "metadata": {"source_path": "/tmp/doc_b.pdf"},
        }
    ]

    row_out = _build_lancedb_rows_from_df(rows)

    assert len(row_out) == 1
    metadata = ast.literal_eval(row_out[0]["metadata"])
    assert metadata["_content_type"] == "chart"
