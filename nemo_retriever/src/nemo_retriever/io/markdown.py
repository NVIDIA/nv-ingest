# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .dataframe import read_dataframe

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


def to_markdown_by_page(results: object) -> dict[str, dict[int, str]]:
    """Render results as markdown grouped by document, then by page."""
    grouped_records = _coerce_documents(results)
    rendered: dict[str, dict[int, str]] = {}

    for document_name, records in grouped_records.items():
        by_page = _pages_for_records(records)
        rendered[document_name] = {
            page_number: _render_page_content(page_content)
            for page_number, page_content in sorted(by_page.items(), key=_page_sort_key)
        }

    return rendered


def to_markdown(results: object) -> dict[str, str]:
    """Render results as one collapsed markdown string per document."""
    rendered: dict[str, str] = {}

    for document_name, pages in to_markdown_by_page(results).items():
        rendered[document_name] = "\n\n".join(page_markdown for page_markdown in pages.values() if page_markdown)

    return rendered


def _coerce_documents(results: object) -> dict[str, list[dict[str, Any]]]:
    grouped_records: dict[str, list[dict[str, Any]]] = {}
    _extend_documents(grouped_records, results)
    return grouped_records


def _extend_documents(
    grouped_records: dict[str, list[dict[str, Any]]],
    results: object,
    explicit_document_name: str | None = None,
) -> None:
    if results is None:
        return

    dataset = getattr(results, "_rd_dataset", None)
    if dataset is not None:
        _extend_documents(grouped_records, dataset, explicit_document_name)
        return

    take_all = getattr(results, "take_all", None)
    if callable(take_all):
        _extend_documents(grouped_records, take_all(), explicit_document_name)
        return

    if isinstance(results, pd.DataFrame):
        _add_records(grouped_records, results.to_dict(orient="records"), explicit_document_name)
        return

    if isinstance(results, Path):
        payload, payload_document_name = _load_results_path(results)
        _extend_documents(grouped_records, payload, explicit_document_name or payload_document_name)
        return

    if isinstance(results, str):
        path = Path(results).expanduser()
        if path.exists():
            payload, payload_document_name = _load_results_path(path)
            _extend_documents(grouped_records, payload, explicit_document_name or payload_document_name)
            return
        raise TypeError("String inputs must point to a saved results file.")

    if isinstance(results, Mapping):
        extracted = _extract_records_from_mapping(results)
        if extracted is not None:
            records, mapping_document_name = extracted
            _add_records(grouped_records, records, explicit_document_name or mapping_document_name)
            return

        for key, value in results.items():
            _extend_documents(grouped_records, value, str(key))
        return

    if isinstance(results, Iterable) and not isinstance(results, (bytes, bytearray)):
        items = list(results)
        if not items:
            return
        if all(isinstance(item, Mapping) and _looks_like_record(item) for item in items):
            _add_records(grouped_records, [dict(item) for item in items], explicit_document_name)
            return
        for item in items:
            _extend_documents(grouped_records, item, explicit_document_name=None)
        return

    raise TypeError(f"Unsupported results type for markdown rendering: {type(results)!r}")


def _load_results_path(path: Path) -> tuple[object, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, _document_name_from_mapping(payload) if isinstance(payload, Mapping) else None
    if suffix == ".jsonl":
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) == 1 and isinstance(lines[0], Mapping):
            payload = lines[0]
            return payload, _document_name_from_mapping(payload)
        return lines, None
    return read_dataframe(path), None


def _extract_records_from_mapping(results: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None] | None:
    for key in _RECORD_LIST_KEYS:
        value = results.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            return [dict(item) for item in value], _document_name_from_mapping(results)
    if _looks_like_record(results):
        return [dict(results)], None
    return None


def _add_records(
    grouped_records: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
    explicit_document_name: str | None = None,
) -> None:
    fallback_document_name = explicit_document_name or _next_unknown_document_name(grouped_records)
    for record in records:
        document_name = explicit_document_name or _document_name_for_record(record) or fallback_document_name
        grouped_records.setdefault(document_name, []).append(record)


def _next_unknown_document_name(grouped_records: Mapping[str, list[dict[str, Any]]]) -> str:
    index = 1
    while f"document_{index}" in grouped_records:
        index += 1
    return f"document_{index}"


def _pages_for_records(records: Iterable[Mapping[str, Any]]) -> dict[int, _PageContent]:
    by_page: dict[int, _PageContent] = defaultdict(_PageContent)

    for record in records:
        if "document_type" in record:
            _collect_primitive_record(by_page, record)
        else:
            _collect_page_record(by_page, record)

    return by_page


def _render_page_content(page_content: _PageContent) -> str:
    return "\n\n".join(_dedupe_blocks(page_content.text_blocks + page_content.sections))


def _document_name_from_mapping(results: Mapping[str, Any]) -> str | None:
    metadata = results.get("metadata")
    source_metadata = _nested_mapping(metadata, "source_metadata") if isinstance(metadata, Mapping) else {}
    custom_content = _nested_mapping(metadata, "custom_content") if isinstance(metadata, Mapping) else {}

    return _normalize_document_name(
        results.get("filename"),
        results.get("source_path"),
        results.get("path"),
        results.get("source_id"),
        source_metadata.get("source_name"),
        source_metadata.get("source_id"),
        custom_content.get("path"),
        custom_content.get("input_pdf"),
        custom_content.get("pdf_path"),
    )


def _document_name_for_record(record: Mapping[str, Any]) -> str | None:
    metadata = _metadata(record)
    source_metadata = _nested_mapping(metadata, "source_metadata")
    custom_content = _nested_mapping(metadata, "custom_content")

    return _normalize_document_name(
        record.get("filename"),
        record.get("source_path"),
        metadata.get("source_path"),
        source_metadata.get("source_name"),
        custom_content.get("path"),
        custom_content.get("input_pdf"),
        custom_content.get("pdf_path"),
        record.get("path"),
        source_metadata.get("source_id"),
        record.get("source_id"),
    )


def _normalize_document_name(*candidates: Any) -> str | None:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip()
        if not normalized:
            continue
        name = Path(normalized).name
        return name or normalized
    return None


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
