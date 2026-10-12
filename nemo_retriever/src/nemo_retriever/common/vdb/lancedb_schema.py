# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LanceDB row construction, schema, and table helpers (canonical copy under ``vdb``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo_retriever.common.vdb.records import embedding_vector, normalize_content_type


def extract_embedding_from_row(
    row: Any,
    *,
    embedding_column: str = "embedding",
) -> Optional[List[float]]:
    """Extract an embedding vector from a row."""
    return embedding_vector(getattr(row, embedding_column, None))


def extract_source_path_and_page(row: Any, *, provenance_page: Optional[int] = None) -> Tuple[str, int]:
    """Best-effort extract of source path and page number from a row.

    In service mode the client pre-splits multi-page PDFs into single pages
    before uploading.  The pipeline therefore always sees a 1-page document
    and sets ``page_number = 1``.  However, the *original* page number from
    the source document is preserved in the ``_page_number`` provenance
    column.  We prefer that value when it exceeds the pipeline-derived one.

    Parameters
    ----------
    provenance_page
        The original page number from the client upload metadata.  Passed
        explicitly because ``_page_number`` (underscore-prefixed) is not
        accessible via ``getattr`` on pandas namedtuples from ``itertuples()``.
    """
    path = ""
    page = -1

    value = getattr(row, "path", None)
    if isinstance(value, str) and value.strip():
        path = value.strip()

    value = getattr(row, "page_number", None)
    try:
        if value is not None:
            page = int(value)
    except Exception:
        pass

    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        source_path = meta.get("source_path")
        if isinstance(source_path, str) and source_path.strip():
            path = source_path.strip()
        content_metadata = meta.get("content_metadata")
        if isinstance(content_metadata, dict) and page == -1:
            hierarchy = content_metadata.get("hierarchy")
            if isinstance(hierarchy, dict) and "page" in hierarchy:
                try:
                    page = int(hierarchy.get("page"))
                except Exception:
                    pass

    # Prefer the provenance page number set by service-mode pre-splitting.
    # In service mode _page_number carries the original page index from the
    # source document; the pipeline's page_number is always 1 since each
    # upload is a single-page PDF.  In batch mode _page_number is 1 (whole
    # document submitted) and the pipeline sets the correct page_number, so
    # taking the max produces the right value in both cases.
    if provenance_page is not None:
        try:
            pp = int(provenance_page)
            if pp > page:
                page = pp
        except (TypeError, ValueError):
            pass

    return path, page


def _build_detection_metadata(row: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    page_element_count = getattr(row, "page_elements_v3_num_detections", None)
    if page_element_count is not None:
        try:
            out["page_elements_v3_num_detections"] = int(page_element_count)
        except Exception:
            pass

    page_element_counts = getattr(row, "page_elements_v3_counts_by_label", None)
    if isinstance(page_element_counts, dict):
        out["page_elements_v3_counts_by_label"] = {
            str(key): int(value)
            for key, value in page_element_counts.items()
            if isinstance(key, str) and value is not None
        }

    for ocr_column in ("table", "chart", "infographic"):
        entries = getattr(row, ocr_column, None)
        if isinstance(entries, list):
            out[f"ocr_{ocr_column}_detections"] = int(len(entries))

    return out


def update_metadata_with_content_type(metadata_obj: Dict[str, Any], *, content_type: Any) -> None:
    normalized = normalize_content_type(content_type)
    if normalized is None:
        return
    metadata_obj["_content_type"] = normalized


def build_lancedb_row(
    row: Any,
    *,
    embedding_column: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
    provenance_page: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build a single LanceDB-ready dict from a DataFrame row."""
    embedding = extract_embedding_from_row(row, embedding_column=embedding_column)
    if embedding is None:
        return None

    path, page_number = extract_source_path_and_page(row, provenance_page=provenance_page)
    path_obj = Path(path) if path else None
    filename = path_obj.name if path_obj is not None else ""
    pdf_basename = path_obj.stem if path_obj is not None else ""
    pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""
    source_id = path or filename or pdf_basename

    metadata_obj: Dict[str, Any] = {"page_number": int(page_number) if page_number is not None else -1}
    if pdf_page:
        metadata_obj["pdf_page"] = pdf_page
    metadata_obj.update(_build_detection_metadata(row))
    update_metadata_with_content_type(metadata_obj, content_type=getattr(row, "_content_type", None))

    orig_meta = getattr(row, "metadata", None)
    if isinstance(orig_meta, dict):
        for key in ("chunk_index", "chunk_count"):
            if key in orig_meta:
                metadata_obj[key] = orig_meta[key]

    source_obj: Dict[str, Any] = {"source_id": str(path)}
    row_out: Dict[str, Any] = {
        "vector": embedding,
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
        text = getattr(row, text_column, None)
        row_out["text"] = str(text) if isinstance(text, str) else ""
    else:
        row_out["text"] = ""

    stored_uri = getattr(row, "_stored_image_uri", None)
    row_out["stored_image_uri"] = str(stored_uri) if stored_uri else ""

    content_type = getattr(row, "_content_type", None)
    row_out["content_type"] = str(content_type) if content_type else ""

    bbox = getattr(row, "_bbox_xyxy_norm", None)
    row_out["bbox_xyxy_norm"] = json.dumps(bbox) if bbox else ""

    return row_out


def build_lancedb_rows(
    df: Any,
    *,
    embedding_column: str = "embedding",
    text_column: str = "text",
    include_text: bool = True,
) -> List[Dict[str, Any]]:
    """Build LanceDB rows from a pandas DataFrame."""
    import pandas as pd

    # Extract _page_number explicitly since itertuples() mangles
    # underscore-prefixed column names into positional _0, _1, etc.
    has_provenance = isinstance(df, pd.DataFrame) and "_page_number" in df.columns
    provenance_pages = df["_page_number"].tolist() if has_provenance else None

    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(df.itertuples(index=False)):
        pp = int(provenance_pages[i]) if provenance_pages is not None else None
        row_out = build_lancedb_row(
            row,
            embedding_column=embedding_column,
            text_column=text_column,
            include_text=include_text,
            provenance_page=pp,
        )
        if row_out is not None:
            rows.append(row_out)
    return rows


def lancedb_schema(vector_dim: int = 2048, *, collection_managed: bool = False) -> Any:
    """Return a PyArrow schema for the standard LanceDB table layout."""
    import pyarrow as pa  # type: ignore

    fields = [
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        pa.field("pdf_page", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("pdf_basename", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("source", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("path", pa.string()),
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("stored_image_uri", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("bbox_xyxy_norm", pa.string()),
    ]
    if collection_managed:
        fields.extend(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("document_version", pa.string()),
                pa.field("content_sha256", pa.string()),
                pa.field("created_at", pa.string()),
            ]
        )
    return pa.schema(fields)


def infer_vector_dim(rows: List[Dict[str, Any]]) -> int:
    """Return the embedding dimension from the first row that has a vector."""
    for row in rows:
        vector = row.get("vector")
        if isinstance(vector, list) and vector:
            return len(vector)
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
