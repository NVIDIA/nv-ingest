# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .dataframe import read_dataframe

_DOCUMENT_TITLE = "Extracted Content"
_UNKNOWN_PAGE = -1
_RECORD_LIST_KEYS = ("records", "df_records", "extracted_df_records", "primitives")
_PAGE_CONTENT_COLUMNS = (
    ("table", "Table"),
    ("chart", "Chart"),
    ("infographic", "Infographic"),
    ("tables", "Table"),
    ("charts", "Chart"),
    ("infographics", "Infographic"),
)


@dataclass
class _PageContent:
    text_blocks: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def next_index(self, label: str) -> int:
        next_value = self.counters.get(label, 0) + 1
        self.counters[label] = next_value
        return next_value


def to_markdown_by_page(results: object) -> dict[int, str]:
    """Render a single document result as markdown grouped by page."""
    records = _coerce_records(results)
    by_page: dict[int, _PageContent] = defaultdict(_PageContent)

    for record in records:
        if "document_type" in record:
            _collect_primitive_record(by_page, record)
        else:
            _collect_page_record(by_page, record)

    rendered: dict[int, str] = {}
    for page_number, page_content in sorted(by_page.items(), key=_page_sort_key):
        blocks = _dedupe_blocks(page_content.text_blocks + page_content.sections)
        header = f"## Page {page_number}" if page_number != _UNKNOWN_PAGE else "## Page Unknown"
        rendered[page_number] = header + ("\n\n" + "\n\n".join(blocks) if blocks else "\n")

    return rendered


def to_markdown(results: object) -> str | None:
    """Render a single document result as one markdown document."""
    pages = to_markdown_by_page(results)
    if not pages:
        return None
    return f"# {_DOCUMENT_TITLE}\n\n" + "\n\n".join(pages.values())


def _coerce_records(results: object) -> list[dict[str, Any]]:
    if results is None:
        return []
    if isinstance(results, pd.DataFrame):
        return results.to_dict(orient="records")
    if isinstance(results, Path):
        return read_dataframe(results).to_dict(orient="records")
    if isinstance(results, str):
        path = Path(results).expanduser()
        if path.exists():
            return read_dataframe(path).to_dict(orient="records")
        raise TypeError("String inputs must point to a saved results file.")
    if isinstance(results, Mapping):
        return _records_from_mapping(results)
    if isinstance(results, Iterable) and not isinstance(results, (bytes, bytearray)):
        return _records_from_iterable(results)
    raise TypeError(f"Unsupported results type for markdown rendering: {type(results)!r}")


def _records_from_iterable(results: Iterable[Any]) -> list[dict[str, Any]]:
    items = list(results)
    if not items:
        return []
    if len(items) == 1:
        first = items[0]
        if not isinstance(first, Mapping):
            return _coerce_records(first)
        if not _looks_like_record(first):
            return _records_from_mapping(first)
    if all(isinstance(item, Mapping) for item in items):
        return [dict(item) for item in items]
    raise ValueError("Markdown rendering expects a single document result. Pass one document, such as results[0].")


def _records_from_mapping(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in _RECORD_LIST_KEYS:
        value = results.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return [dict(item) for item in value]
    if _looks_like_record(results):
        return [dict(results)]
    raise ValueError("Markdown rendering expects a document row, row list, or saved results payload.")

    # TODO(jioffe): Centralize retriever result-shape detection so helpers and
    # actor outputs do not rely on duplicated key-based heuristics.


def _looks_like_record(record: Mapping[str, Any]) -> bool:
    return any(
        key in record
        for key in (
            "document_type",
            "page_number",
            "text",
            "content",
            "metadata",
            "table",
            "chart",
            "infographic",
            "tables",
            "charts",
            "infographics",
        )
    )


def _collect_page_record(by_page: dict[int, _PageContent], record: Mapping[str, Any]) -> None:
    page_number = _page_number_for_record(record)
    _append_text(
        by_page[page_number], _first_text(record.get("text"), record.get("content"), _metadata_content(record))
    )

    for column_name, label in _PAGE_CONTENT_COLUMNS:
        items = record.get(column_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            _append_section(
                by_page[page_number],
                label,
                _first_text(item.get("text"), item.get("table_content"), item.get("content")),
            )


def _collect_primitive_record(by_page: dict[int, _PageContent], record: Mapping[str, Any]) -> None:
    page_number = _page_number_for_record(record)
    metadata = _metadata(record)
    content_metadata = _content_metadata(metadata)
    document_type = record.get("document_type")

    if document_type == "text":
        _append_text(
            by_page[page_number],
            _first_text(record.get("text"), metadata.get("content"), record.get("content")),
        )
        return

    if document_type == "structured":
        label = _label_for_subtype(content_metadata.get("subtype"), fallback="Structured")
        table_metadata = _nested_mapping(metadata, "table_metadata")
        _append_section(
            by_page[page_number],
            label,
            _first_text(table_metadata.get("table_content"), metadata.get("content"), record.get("text")),
        )
        return

    if document_type == "image":
        subtype = content_metadata.get("subtype")
        label = "Page Image" if subtype == "page_image" else "Image"
        image_metadata = _nested_mapping(metadata, "image_metadata")
        _append_section(
            by_page[page_number],
            label,
            _first_text(image_metadata.get("text"), image_metadata.get("caption"), metadata.get("content")),
        )
        return

    if document_type == "audio":
        audio_metadata = _nested_mapping(metadata, "audio_metadata")
        _append_section(by_page[page_number], "Audio", _first_text(audio_metadata.get("audio_transcript")))
        return

    _append_text(by_page[page_number], _first_text(record.get("text"), metadata.get("content"), record.get("content")))


def _append_text(page_content: _PageContent, text: str | None) -> None:
    if text is not None:
        page_content.text_blocks.append(text)


def _append_section(page_content: _PageContent, label: str, text: str | None) -> None:
    if text is None:
        return
    page_content.sections.append(f"### {label} {page_content.next_index(label)}\n\n{text}")


def _dedupe_blocks(blocks: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for block in blocks:
        normalized = block.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _page_number_for_record(record: Mapping[str, Any]) -> int:
    page_number = _safe_page_number(record.get("page_number"))
    if page_number != _UNKNOWN_PAGE:
        return page_number

    metadata = _metadata(record)
    content_metadata = _content_metadata(metadata)
    page_number = _safe_page_number(content_metadata.get("page_number"))
    if page_number != _UNKNOWN_PAGE:
        return page_number

    hierarchy = _nested_mapping(content_metadata, "hierarchy")
    return _safe_page_number(hierarchy.get("page"))


def _safe_page_number(value: Any) -> int:
    try:
        page_number = int(value)
    except (TypeError, ValueError):
        return _UNKNOWN_PAGE
    return page_number if page_number > 0 else _UNKNOWN_PAGE


def _page_sort_key(item: tuple[int, _PageContent]) -> tuple[int, int]:
    page_number = item[0]
    if page_number == _UNKNOWN_PAGE:
        return (1, 0)
    return (0, page_number)


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _content_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return _nested_mapping(metadata, "content_metadata")


def _nested_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _metadata_content(record: Mapping[str, Any]) -> Any:
    return _metadata(record).get("content")


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _label_for_subtype(subtype: Any, *, fallback: str) -> str:
    if not isinstance(subtype, str):
        return fallback
    subtype_map = {
        "table": "Table",
        "chart": "Chart",
        "infographic": "Infographic",
        "page_image": "Page Image",
    }
    return subtype_map.get(subtype.strip().lower(), fallback)
