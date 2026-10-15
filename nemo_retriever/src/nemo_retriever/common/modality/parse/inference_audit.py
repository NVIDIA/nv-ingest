# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize completed Parse inference independently of searchable text rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math


def summarize_parse_inference(result: object) -> dict[str, int]:
    """Count unique source pages, conservatively combining split/duplicate rows.

    Only diagnostic and identity columns are inspected. The already returned
    result is used; this function never runs the graph or reads the VDB again.
    Missing identities and old payloads cannot manufacture successful coverage.
    """
    columns = getattr(result, "columns", ())
    selected = [
        key
        for key in ("path", "source_path", "document_id", "source_id", "page_number", "metadata", "nemotron_parse_v1_2")
        if key in columns
    ]
    rows = result.loc[:, selected].to_dict("records") if selected else []
    pages: dict[tuple[str, str], str] = {}
    priority = {"succeeded": 0, "skipped": 1, "missing_evidence": 2, "failed": 3}
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_metadata = metadata.get("source_metadata") or {}
        source = next(
            (
                row.get(key)
                for key in ("path", "source_path", "document_id", "source_id")
                if isinstance(row.get(key), str) and row[key]
            ),
            None,
        )
        source = source or metadata.get("source_path") or source_metadata.get("source_id")
        page = row.get("page_number")
        if page is None:
            page = (metadata.get("content_metadata") or {}).get("page_number")
        if page is None:
            page = metadata.get("page_number")
        valid_page = (
            isinstance(page, (int, float))
            and not isinstance(page, bool)
            and math.isfinite(page)
            and page >= 1
            and int(page) == page
        )
        identity_valid = isinstance(source, str) and bool(source) and valid_page
        key = (source, str(int(page))) if identity_valid else ("missing-identity", str(index))
        evidence = row.get("nemotron_parse_v1_2")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        status = evidence.get("inference_status", "missing_evidence")
        if evidence.get("error"):
            status = "failed"
        elif not identity_valid or status not in priority:
            status = "missing_evidence"
        if key not in pages or priority[status] > priority[pages[key]]:
            pages[key] = status
    counts = Counter(pages.values())
    return {"pages_seen": len(pages), **{f"pages_{status}": counts[status] for status in priority}}
