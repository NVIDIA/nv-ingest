# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict Parse page projection into the interleaved Parquet schema.

Page validation and element projection follow the PDF recipe's publication
contract. A disk spool preserves document/page ordering even when graph batches
arrive out of order, without retaining the extracted corpus in memory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

INTERLEAVED_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string()),
        pa.field("text_content", pa.string()),
        pa.field("binary_content", pa.large_binary()),
        pa.field("source_ref", pa.string()),
        pa.field("materialize_error", pa.string()),
    ]
)
PROJECTION_SCHEMA = pa.schema(
    list(INTERLEAVED_SCHEMA)
    + [
        pa.field("page_number", pa.int32()),
        pa.field("source_path", pa.string()),
        pa.field("element_class", pa.string()),
        pa.field("bbox_xyxy_norm", pa.list_(pa.float64())),
        pa.field("bbox_coordinate_space", pa.string()),
    ]
)
MIN_PICTURE_CROP_PX = 10
_NEMOTRON_ELEMENT_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)" r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)
_STRUCTURAL_TOKEN_RE = re.compile(
    r"<[xy]_0\.\d+>|<class_(?:Bibliography|Caption|Code|Footnote|Formula|List-item|Page-footer|Page-header"
    r"|Picture|Section-header|TOC|Table|Text|Title)>"
)


class IncompleteModelOutputError(ValueError):
    """The tagged response contains content outside complete elements."""


def _validate_complete_raw_output(raw_output: str) -> int:
    """Return the number of elements only when the entire response parses."""

    if not isinstance(raw_output, str):
        msg = f"raw model output must be text, got {type(raw_output).__name__}"
        raise TypeError(msg)
    matches = list(_NEMOTRON_ELEMENT_RE.finditer(raw_output))
    cursor = 0
    for match in matches:
        if raw_output[cursor : match.start()].strip():
            msg = f"raw model output has unparsed content at offset {cursor}"
            raise IncompleteModelOutputError(msg)
        if _STRUCTURAL_TOKEN_RE.search(match.group(3)):
            msg = "raw model output contains incomplete or unbalanced element tags"
            raise IncompleteModelOutputError(msg)
        cursor = match.end()
    if raw_output[cursor:].strip():
        msg = f"raw model output has unparsed content at offset {cursor}"
        raise IncompleteModelOutputError(msg)
    return len(matches)


def _parse_raw_elements(raw_output: str) -> list[dict[str, Any]]:
    expected_count = _validate_complete_raw_output(raw_output)

    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import (
        extract_classes_bboxes,
        postprocess_text,
    )

    classes, bboxes, texts = extract_classes_bboxes(raw_output)
    if not (len(classes) == len(bboxes) == len(texts) == expected_count):
        msg = "NRL parser did not return every validated model element"
        raise IncompleteModelOutputError(msg)

    elements: list[dict[str, Any]] = []
    for element_class, bbox, text in zip(classes, bboxes, texts, strict=True):
        processed = postprocess_text(
            text,
            cls=element_class,
            text_format="markdown",
            table_format="latex",  # Preserve native table bodies and merged-cell spans for Curator.
            blank_text_in_figures=False,
        ).strip()
        elements.append({"class": element_class, "text": processed, "bbox": list(bbox)})
    return elements


def _crop_picture_bytes(page_image: object, bbox: Sequence[float], *, coordinate_space: str) -> bytes | None:
    from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox

    if not isinstance(page_image, dict) or not isinstance(page_image.get("image_b64"), str):
        return None
    shape = page_image.get("orig_shape_hw")
    tolist = getattr(shape, "tolist", None)
    if callable(tolist):
        shape = tolist()
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        height, width = (int(item) for item in shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if height <= 0 or width <= 0:
        return None

    if coordinate_space == "normalized_page_content":
        from nemotron_parse.postprocessing import transform_bbox_to_original
    else:
        from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import transform_bbox_to_original

    left, top, right, bottom = transform_bbox_to_original(tuple(float(item) for item in bbox), width, height)
    normalized = [left / width, top / height, right / width, bottom / height]
    cropped_b64, cropped_shape = _crop_b64_image_by_norm_bbox(
        page_image["image_b64"],
        bbox_xyxy_norm=normalized,
        image_format="png",
    )
    if cropped_b64 is None or cropped_shape is None or min(cropped_shape) < MIN_PICTURE_CROP_PX:
        return None
    try:
        payload = base64.b64decode(cropped_b64, validate=True)
    except (TypeError, ValueError):
        return None
    return payload or None


def _normalize_bbox(value: object) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(coordinate, bool) for coordinate in value):
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return None
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        return None
    return bbox


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def project_page(row: Mapping[str, Any], passthrough_columns: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Keep a whole valid page, or one explicit error row; never salvage partial tags."""
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    source = str(row.get("path") or metadata.get("source_path") or row.get("source_id") or "")
    if not source:
        raise ValueError("Interleaved projection requires path, metadata.source_path or source_id")
    sample_id = str(row.get("source_id") or hashlib.sha256(source.encode()).hexdigest())
    page = row.get("page_number")
    native_page = (
        int(page)
        if isinstance(page, (int, float)) and not isinstance(page, bool) and math.isfinite(page) and int(page) == page
        else 0
    )
    parser = row.get("nemotron_parse_v1_2")
    parser = parser if isinstance(parser, Mapping) else {}
    coordinate_space = parser.get("bbox_coordinate_space") or "normalized_1664x2048_padded_canvas"
    base = {
        "sample_id": sample_id,
        "position": 0,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": None,
        "binary_content": None,
        "source_ref": _json({"path": source}),
        "materialize_error": None,
        "page_number": native_page - 1 if native_page > 0 else None,
        "source_path": source,
        "element_class": None,
        "bbox_xyxy_norm": None,
        "bbox_coordinate_space": coordinate_space,
        **{name: row.get(name) for name in passthrough_columns},
    }
    error = metadata.get("error") or parser.get("error") or row.get("error")
    if error:
        return [{**base, "materialize_error": _json(error)}]
    try:
        if native_page < 1:
            raise ValueError("Missing or invalid one-based page_number")
        raw = parser.get("routing_output", parser.get("raw_output"))
        if not isinstance(raw, str):
            raise ValueError("Missing tagged Nemotron Parse output")
        reason = parser.get("finish_reason")
        if reason and str(reason).lower() not in {"stop", "eos"}:
            raise IncompleteModelOutputError(f"Model ended with finish_reason={reason!r}")
        if not raw.strip():
            raise IncompleteModelOutputError("Empty Parse output; blank-source correctness has not been established")
        elements = _parse_raw_elements(raw)
        projected = []
        for element in elements:
            bbox = _normalize_bbox(element["bbox"])
            if bbox is None or not element["class"]:
                raise ValueError("Invalid element box or class")
            picture = element["class"] == "Picture"
            binary = (
                _crop_picture_bytes(row.get("page_image"), bbox, coordinate_space=coordinate_space) if picture else None
            )
            if picture and binary is None:
                raise ValueError("Picture crop could not be decoded or is too small")
            projected.append(
                {
                    **base,
                    "position": len(projected),
                    "modality": "image" if picture else "table" if element["class"] == "Table" else "text",
                    "content_type": "image/png" if picture else "text/markdown",
                    "text_content": element["text"],
                    "binary_content": binary,
                    "element_class": element["class"],
                    "bbox_xyxy_norm": bbox,
                }
            )
        return projected
    except (TypeError, ValueError, OSError) as exc:
        return [
            {
                **base,
                "materialize_error": _json(
                    {"stage": "interleaved_projection", "type": type(exc).__name__, "message": str(exc)}
                ),
            }
        ]


def projection_table(rows: list[dict[str, Any]]) -> pa.Table:
    inferred = pa.Table.from_pylist(rows)
    fields = list(PROJECTION_SCHEMA) + [f for f in inferred.schema if f.name not in PROJECTION_SCHEMA.names]
    return pa.Table.from_pylist(rows, schema=pa.schema(fields))


class InterleavedSpool:
    """Disk-backed document ordering; at most one projected page is decoded at once."""

    def __init__(self, directory: Path, passthrough_columns: Sequence[str]):
        self.path = directory / "_projection.sqlite"
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TABLE pages (sample TEXT, page INTEGER, source TEXT, failed INTEGER, data BLOB, PRIMARY KEY(sample,page))"
        )
        self.passthrough_columns = passthrough_columns

    def add(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            elements = project_page(row, self.passthrough_columns)
            first = elements[0]
            table = projection_table(elements)
            buffer = pa.BufferOutputStream()
            with pa.ipc.new_stream(buffer, table.schema) as writer:
                writer.write_table(table)
            page = first["page_number"]
            try:
                self.connection.execute(
                    "INSERT INTO pages VALUES(?,?,?,?,?)",
                    (
                        first["sample_id"],
                        -1 if page is None else page,
                        first["source_path"],
                        int(first["materialize_error"] is not None),
                        buffer.getvalue().to_pybytes(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate page identity: {first['sample_id']} page {page}") from exc
        self.connection.commit()

    def iter_rows(self):
        position = 0
        counts = self.connection.execute(
            "SELECT sample, source, COUNT(*), SUM(failed) FROM pages GROUP BY sample ORDER BY sample"
        )
        for sample, source, pages, failed in counts:
            yield {
                "sample_id": sample,
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "text_content": _json(
                    {
                        "source_path": source,
                        "observed_pages": pages,
                        "failed_pages": failed,
                        "extraction_status": "failed" if failed == pages else "partial" if failed else "success",
                    }
                ),
                "binary_content": None,
                "source_ref": _json({"path": source}),
                "materialize_error": None,
                "page_number": None,
                "source_path": source,
                "element_class": None,
                "bbox_xyxy_norm": None,
                "bbox_coordinate_space": None,
            }
            position = 0
            for (data,) in self.connection.execute("SELECT data FROM pages WHERE sample=? ORDER BY page", (sample,)):
                for row in pa.ipc.open_stream(data).read_all().to_pylist():
                    row["position"] = position
                    position += 1
                    yield row

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)
