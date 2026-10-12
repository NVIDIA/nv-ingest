# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Serialize and deserialize ingest pipeline DataFrames for service transport.

The retriever service returns per-document rows over HTTP; these helpers
keep the wire format aligned with the ``pandas.DataFrame`` produced by
:meth:`nemo_retriever.ingestor.graph_ingestor.GraphIngestor.ingest` in
``inprocess`` and ``batch`` run modes (same column names and row shape),
while stripping bulky raw images and embeddings from cell values. Callers
can opt into the future compact result schema with ``result_schema="compact"``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

ResultSchema = Literal["legacy", "compact"]

_DEFAULT_EMBEDDING_COLUMN = "embedding"
_MAX_STR_LEN = 500
_TEXT_FIELD_NAMES = frozenset({"caption", "content", "text"})
_RAW_IMAGE_FIELD_NAMES = frozenset({"image_b64", "_image_b64"})
_EMBEDDING_FIELD_NAMES = frozenset({"embedding", "embeddings"})
_ELEMENT_TYPE_ALIASES = {
    "chart_caption": "chart",
    "image_caption": "image",
    "images": "image",
    "infographic_caption": "infographic",
    "table_caption": "table",
    "text/page": "text",
}
_MEDIA_ELEMENT_TYPES = frozenset({"audio", "video", "video_frame"})


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (dict, list, tuple, bytes, bytearray, memoryview, np.ndarray)):
        return False
    try:
        import pandas as pd

        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def sanitize_cell_value(val: Any) -> Any:
    """Convert a single cell value to a JSON-safe, memory-friendly form."""
    if _is_missing(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return f"<ndarray shape={val.shape} dtype={val.dtype}>"
    if isinstance(val, (list, tuple)) and len(val) > 20:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, bytes):
        return f"<bytes len={len(val)}>"
    if isinstance(val, str) and len(val) > _MAX_STR_LEN:
        return val[:_MAX_STR_LEN] + f"…[{len(val)} chars total]"
    return val


def _sanitize_returned_payload(val: Any) -> Any:
    """Convert explicitly requested bulky payloads without size summarization."""
    if _is_missing(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return _sanitize_returned_payload(val.tolist())
    if isinstance(val, dict):
        return {str(k): _sanitize_returned_payload(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_returned_payload(item) for item in val]
    if isinstance(val, str):
        return val
    return sanitize_cell_value(val)


def _contains_requested_payload(val: Any, *, return_embeddings: bool, return_images: bool) -> bool:
    if isinstance(val, dict):
        for key, nested in val.items():
            str_key = str(key)
            if return_images and str_key in _RAW_IMAGE_FIELD_NAMES:
                return True
            if return_embeddings and str_key in _EMBEDDING_FIELD_NAMES:
                return True
            if _contains_requested_payload(
                nested,
                return_embeddings=return_embeddings,
                return_images=return_images,
            ):
                return True
        return False
    if isinstance(val, (list, tuple)):
        return any(
            _contains_requested_payload(
                item,
                return_embeddings=return_embeddings,
                return_images=return_images,
            )
            for item in val
        )
    return False


def _contains_full_text(val: Any) -> bool:
    if isinstance(val, dict):
        return any(
            (str(key) in _TEXT_FIELD_NAMES and isinstance(nested, str)) or _contains_full_text(nested)
            for key, nested in val.items()
        )
    if isinstance(val, (list, tuple)):
        return any(_contains_full_text(item) for item in val)
    return False


def _sanitize_result_value(
    key: str,
    val: Any,
    *,
    return_embeddings: bool = False,
    return_images: bool = False,
    embedding_column: str = _DEFAULT_EMBEDDING_COLUMN,
) -> Any:
    """Convert a result value to JSON-safe transport form.

    Raw image and embedding payloads are useful inside the pipeline for
    visual embedding, OCR, image storage, and vector DB upload, but returning
    them in service results dominates memory use. Keep the surrounding
    keys/columns stable and null only the bulky payload values.
    """
    if key in _RAW_IMAGE_FIELD_NAMES:
        return _sanitize_returned_payload(val) if return_images else None
    if key in _EMBEDDING_FIELD_NAMES or key == embedding_column:
        return _sanitize_returned_payload(val) if return_embeddings else None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return {
            str(k): _sanitize_result_value(
                str(k),
                v,
                return_embeddings=return_embeddings,
                return_images=return_images,
            )
            for k, v in val.items()
        }
    if isinstance(val, list):
        sanitized = [
            _sanitize_result_value(
                "",
                item,
                return_embeddings=return_embeddings,
                return_images=return_images,
            )
            for item in val
        ]
        if _contains_full_text(val) or _contains_requested_payload(
            val, return_embeddings=return_embeddings, return_images=return_images
        ):
            return sanitized
        return sanitize_cell_value(sanitized)
    if isinstance(val, tuple):
        sanitized = [
            _sanitize_result_value(
                "",
                item,
                return_embeddings=return_embeddings,
                return_images=return_images,
            )
            for item in val
        ]
        if _contains_full_text(val) or _contains_requested_payload(
            val, return_embeddings=return_embeddings, return_images=return_images
        ):
            return sanitized
        return sanitize_cell_value(sanitized)
    return sanitize_cell_value(val)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _content_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return _mapping(metadata.get("content_metadata"))


def _coerce_str(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value)
    return text if text.strip() else None


def _coerce_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        text = _coerce_str(value)
        if text is not None:
            return text
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _coerce_int(value)
        if number is not None:
            return number
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = _coerce_float(value)
        if number is not None:
            return number
    return None


def _normalize_element_type(value: Any) -> str | None:
    raw = _coerce_str(value)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    return _ELEMENT_TYPE_ALIASES.get(normalized, normalized)


def _extract_text(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    return _first_str(row.get("text"), row.get("content"), metadata.get("content")) or ""


def _extract_source_id(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    source = _mapping(row.get("source"))
    return (
        _first_str(
            row.get("source_id"),
            row.get("source_path"),
            metadata.get("source_id"),
            metadata.get("source_path"),
            source.get("source_id"),
            row.get("path"),
            row.get("source"),
        )
        or ""
    )


def _extract_element_type(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    content_metadata = _content_metadata(metadata)
    return (
        _normalize_element_type(
            row.get("element_type")
            or row.get("_content_type")
            or row.get("content_type")
            or row.get("document_type")
            or metadata.get("_content_type")
            or metadata.get("content_type")
            or metadata.get("document_type")
            or content_metadata.get("type")
        )
        or "text"
    )


def _extract_page_number(row: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    content_metadata = _content_metadata(metadata)
    return _first_int(row.get("page_number"), metadata.get("page_number"), content_metadata.get("page_number"))


def _extract_time_bounds(row: dict[str, Any], metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    content_metadata = _content_metadata(metadata)
    start = _first_float(
        row.get("start_time_seconds"),
        row.get("segment_start_seconds"),
        row.get("frame_timestamp_seconds"),
        metadata.get("start_time_seconds"),
        metadata.get("segment_start_seconds"),
        metadata.get("frame_timestamp_seconds"),
        content_metadata.get("start_time_seconds"),
        content_metadata.get("start_time"),
    )
    end = _first_float(
        row.get("end_time_seconds"),
        row.get("segment_end_seconds"),
        metadata.get("end_time_seconds"),
        metadata.get("segment_end_seconds"),
        content_metadata.get("end_time_seconds"),
        content_metadata.get("end_time"),
    )
    return start, end


def _extract_duration(
    row: dict[str, Any], metadata: dict[str, Any], start: float | None, end: float | None
) -> float | None:
    if start is not None and end is not None:
        return max(0.0, float(end) - float(start))
    return _first_float(
        row.get("duration_seconds"), row.get("duration"), metadata.get("duration_seconds"), metadata.get("duration")
    )


def _extract_stored_image_uri(row: dict[str, Any]) -> str | None:
    page_image = row.get("page_image")
    page_uri = page_image.get("stored_image_uri") if isinstance(page_image, dict) else None
    return _first_str(row.get("_stored_image_uri"), row.get("stored_image_uri"), page_uri)


def _extract_embedding(row: dict[str, Any], *, embedding_column: str) -> Any:
    embedding = row.get(embedding_column)
    if _is_missing(embedding):
        return None
    return _sanitize_returned_payload(embedding)


def _compact_error(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        out = {
            str(key): sanitize_cell_value(value[key])
            for key in ("stage", "type", "message")
            if key in value and not _is_missing(value[key])
        }
        return out or None
    return sanitize_cell_value(value)


def compact_result_record(
    row: dict[str, Any],
    *,
    return_embeddings: bool = False,
    embedding_column: str = _DEFAULT_EMBEDDING_COLUMN,
) -> dict[str, Any]:
    """Project a full pipeline row into the compact public result shape.

    Parameters
    ----------
    row
        Full pipeline result row to project.
    return_embeddings
        Include a normalized top-level embedding when the row contains one.
    embedding_column
        Pipeline result column containing the embedding vector.

    Returns
    -------
    dict[str, Any]
        Compact public result record.
    """
    metadata = _mapping(row.get("metadata"))
    element_type = _extract_element_type(row, metadata)
    start, end = _extract_time_bounds(row, metadata)
    duration = _extract_duration(row, metadata, start, end)
    is_media = element_type in _MEDIA_ELEMENT_TYPES or start is not None or end is not None

    record: dict[str, Any] = {
        "text": _extract_text(row, metadata),
        "source_id": _extract_source_id(row, metadata),
        "element_type": element_type,
    }

    if is_media:
        if start is not None:
            record["start_time_seconds"] = start
        if end is not None:
            record["end_time_seconds"] = end
        if duration is not None:
            record["duration_seconds"] = duration
    else:
        page_number = _extract_page_number(row, metadata)
        if page_number is not None:
            record["page_number"] = page_number

    stored_image_uri = _extract_stored_image_uri(row)
    if stored_image_uri is not None:
        record["stored_image_uri"] = stored_image_uri

    if return_embeddings:
        embedding = _extract_embedding(row, embedding_column=embedding_column)
        if embedding is not None:
            record["embedding"] = embedding

    error = _compact_error(row.get("error") if "error" in row else metadata.get("error"))
    if error is not None:
        record["error"] = error

    return record


def dataframe_to_transport_records(
    df: Any,
    *,
    result_schema: ResultSchema = "legacy",
    return_embeddings: bool = False,
    return_images: bool = False,
    embedding_column: str | None = None,
) -> list[dict[str, Any]]:
    """Serialize a pipeline DataFrame to JSON-safe row dicts.

    ``result_schema="legacy"`` retains all columns and complete string
    values so the reconstructed frame matches ``GraphIngestor.ingest()``
    output; by default raw image and embedding payload values are nulled
    before transport.

    ``result_schema="compact"`` returns the future compact public schema:
    extracted text, source provenance, element type, page number or media
    timings, optional stored-image URIs, optional embeddings, and optional
    errors.
    """
    import pandas as pd

    embedding_column = embedding_column or _DEFAULT_EMBEDDING_COLUMN
    if result_schema not in ("legacy", "compact"):
        raise ValueError(f"unknown result_schema: {result_schema!r}")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
    records = df.to_dict(orient="records")
    if result_schema == "compact":
        return [
            compact_result_record(
                row,
                return_embeddings=return_embeddings,
                embedding_column=embedding_column,
            )
            for row in records
        ]
    return [
        {
            k: _sanitize_result_value(
                str(k),
                v,
                return_embeddings=return_embeddings,
                return_images=return_images,
                embedding_column=embedding_column,
            )
            for k, v in row.items()
        }
        for row in records
    ]


def dataframe_from_transport_records(records: list[dict[str, Any]]) -> Any:
    """Rebuild a pipeline DataFrame from transport row dicts."""
    import pandas as pd

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def concat_ingest_results(
    rows_by_document: dict[str, list[dict[str, Any]]],
    document_order: list[str],
) -> Any:
    """Concatenate per-document transport rows in upload order.

    Mirrors how :class:`~nemo_retriever.graph.executor.InprocessExecutor`
    processes a list of input paths as one combined result frame.
    """
    import pandas as pd

    frames: list[pd.DataFrame] = []
    for doc_id in document_order:
        rows = rows_by_document.get(doc_id)
        if rows:
            frames.append(dataframe_from_transport_records(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
