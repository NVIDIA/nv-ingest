import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row, build_lancedb_rows


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


def test_build_lancedb_rows_from_arrow_df_skips_null_embeddings() -> None:
    frame = pd.DataFrame(
        {
            "path": ["/tmp/doc_b.pdf", "/tmp/doc_c.pdf"],
            "page_number": [3, 4],
            "text": ["chart text", "missing vector"],
            "_content_type": ["chart_caption", "text"],
            "embedding": pd.Series(
                [[0.5, 0.25], None],
                dtype=pd.ArrowDtype(pa.list_(pa.float32())),
            ),
        }
    )

    row_out = build_lancedb_rows(frame)

    assert len(row_out) == 1
    assert row_out[0]["vector"] == [0.5, 0.25]


def test_build_lancedb_rows_reads_non_identifier_embedding_column() -> None:
    frame = pd.DataFrame(
        {
            "path": ["/tmp/doc_d.pdf", "/tmp/doc_e.pdf"],
            "page_number": [1, 2],
            "text": ["first", "missing vector"],
            "my-vector": pd.Series(
                [[1.0, 0.0], None],
                dtype=pd.ArrowDtype(pa.list_(pa.float32())),
            ),
        }
    )

    row_out = build_lancedb_rows(frame, embedding_column="my-vector")

    assert len(row_out) == 1
    assert row_out[0]["vector"] == [1.0, 0.0]


def test_build_lancedb_rows_uses_first_duplicate_embedding_column() -> None:
    frame = pd.DataFrame([["/tmp/doc_f.pdf", [1.0, 0.0], [0.0, 1.0]]], columns=["path", "embedding", "embedding"])

    row_out = build_lancedb_rows(frame)

    assert len(row_out) == 1
    assert row_out[0]["vector"] == [1.0, 0.0]
