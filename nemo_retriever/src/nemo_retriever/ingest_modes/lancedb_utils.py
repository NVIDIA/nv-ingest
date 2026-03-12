# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared LanceDB row construction, schema, and table helpers.

Consolidates the duplicated logic that previously lived independently in
``inprocess.py`` (``upload_embeddings_to_lancedb_inprocess``) and
``batch.py`` (``_LanceDBWriteActor._build_rows``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_embedding_from_row(
    row: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
) -> Optional[List[float]]:
    """Extract an embedding vector from a row (namedtuple or pd.Series).

    Supports:
    - ``metadata.embedding`` (preferred if present)
    - *embedding_column* payloads like ``{"embedding": [...], ...}``
    """
    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]

    payload = getattr(row, embedding_column, None)
    if isinstance(payload, dict):
        emb = payload.get(embedding_key)
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]
    return None


def extract_source_path_and_page(row: Any) -> Tuple[str, int]:
    """Best-effort extract of source path and page number from a row."""
    path = ""
    page = -1

    v = getattr(row, "path", None)
    if isinstance(v, str) and v.strip():
        path = v.strip()

    v = getattr(row, "page_number", None)
    try:
        if v is not None:
            page = int(v)
    except Exception:
        pass

    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        sp = meta.get("source_path")
        if isinstance(sp, str) and sp.strip():
            path = sp.strip()
        cm = meta.get("content_metadata")
        if isinstance(cm, dict) and page == -1:
            h = cm.get("hierarchy")
            if isinstance(h, dict) and "page" in h:
                try:
                    page = int(h.get("page"))
                except Exception:
                    pass

    return path, page


def _build_detection_metadata(row: Any) -> Dict[str, Any]:
    """Extract per-page detection counters from a row for LanceDB metadata."""
    out: Dict[str, Any] = {}

    pe_num = getattr(row, "page_elements_v3_num_detections", None)
    if pe_num is not None:
        try:
            out["page_elements_v3_num_detections"] = int(pe_num)
        except Exception:
            pass

    pe_counts = getattr(row, "page_elements_v3_counts_by_label", None)
    if isinstance(pe_counts, dict):
        out["page_elements_v3_counts_by_label"] = {
            str(k): int(v) for k, v in pe_counts.items() if isinstance(k, str) and v is not None
        }

    for ocr_col in ("table", "chart", "infographic"):
        entries = getattr(row, ocr_col, None)
        if isinstance(entries, list):
            out[f"ocr_{ocr_col}_detections"] = int(len(entries))

    return out


def build_lancedb_row(
    row: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a single LanceDB-ready dict from a DataFrame row.

    Returns ``None`` when no embedding is found in the row.
    """
    emb = extract_embedding_from_row(row, embedding_column=embedding_column, embedding_key=embedding_key)
    if emb is None:
        return None

    path, page_number = extract_source_path_and_page(row)
    p = Path(path) if path else None
    filename = p.name if p is not None else ""
    pdf_basename = p.stem if p is not None else ""
    pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""
    source_id = path or filename or pdf_basename

    metadata_obj: Dict[str, Any] = {"page_number": int(page_number) if page_number is not None else -1}
    if pdf_page:
        metadata_obj["pdf_page"] = pdf_page
    metadata_obj.update(_build_detection_metadata(row))

    # Preserve split metadata (chunk_index, chunk_count) from the original row.
    orig_meta = getattr(row, "metadata", None)
    if isinstance(orig_meta, dict):
        for k in ("chunk_index", "chunk_count"):
            if k in orig_meta:
                metadata_obj[k] = orig_meta[k]

    source_obj: Dict[str, Any] = {"source_id": str(path)}

    row_out: Dict[str, Any] = {
        "vector": emb,
        "pdf_page": pdf_page,
        "filename": filename,
        "pdf_basename": pdf_basename,
        "page_number": int(page_number) if page_number is not None else -1,
        "source_id": str(source_id),
        "path": str(path),
        "metadata": json.dumps(metadata_obj, ensure_ascii=False),
        "source": json.dumps(source_obj, ensure_ascii=False),
    }

    if include_text:
        t = getattr(row, text_column, None)
        row_out["text"] = str(t) if isinstance(t, str) else ""
    else:
        row_out["text"] = ""

    return row_out


def build_lancedb_rows(
    df: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Build LanceDB rows from a pandas DataFrame.

    Iterates with ``itertuples`` and delegates to :func:`build_lancedb_row`.
    Rows without an embedding are silently skipped.
    """
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        row_out = build_lancedb_row(
            r,
            embedding_column=embedding_column,
            embedding_key=embedding_key,
            text_column=text_column,
            include_text=include_text,
        )
        if row_out is not None:
            rows.append(row_out)
    return rows


def lancedb_schema(vector_dim: int = 2048) -> Any:
    """Return a PyArrow schema for the standard LanceDB table layout."""
    import pyarrow as pa  # type: ignore

    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source", pa.string()),
            pa.field(
                "source_id", pa.string()
            ),  # Different than the source. Field contains path+page_number for aggregation tasks
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
        ]
    )


def infer_vector_dim(rows: List[Dict[str, Any]]) -> int:
    """Return the embedding dimension from the first row that has a vector."""
    for r in rows:
        v = r.get("vector")
        if isinstance(v, list) and v:
            return len(v)
    return 0


def create_or_append_lancedb_table(
    db: Any,
    table_name: str,
    rows: List[Dict[str, Any]],
    schema: Any,
    overwrite: bool = True,
) -> Any:
    """Create or append to a LanceDB table, returning the table object."""
    if overwrite:
        return db.create_table(str(table_name), data=list(rows), schema=schema, mode="overwrite")

    try:
        table = db.open_table(str(table_name))
        table.add(list(rows))
        return table
    except Exception:
        return db.create_table(str(table_name), data=list(rows), schema=schema, mode="create")
