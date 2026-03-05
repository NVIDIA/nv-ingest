# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from nemo_retriever.harness.config import VALID_RECALL_ADAPTERS


def _normalize_pdf_name(value: object) -> str:
    return str(value).replace(".pdf", "")


def _adapt_page_plus_one(query_csv: Path, output_csv: Path) -> Path:
    df = pd.read_csv(query_csv)
    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})

    required = {"query", "pdf", "page"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "page_plus_one adapter requires ['query','pdf','page'] columns "
            f"(missing: {sorted(missing)}) in {query_csv}"
        )

    page_numbers = pd.to_numeric(df["page"], errors="raise").astype(int) + 1
    normalized = pd.DataFrame(
        {
            "query": df["query"].astype(str),
            "pdf_page": [_normalize_pdf_name(pdf) + f"_{page}" for pdf, page in zip(df["pdf"], page_numbers)],
        }
    )
    normalized.to_csv(output_csv, index=False)
    return output_csv


def _adapt_financebench_json(query_json: Path, output_csv: Path) -> Path:
    payload = json.loads(query_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"financebench_json adapter expects a JSON list in {query_json}")

    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        contexts = item.get("contexts")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(contexts, list) or not contexts:
            continue
        context0 = contexts[0]
        if not isinstance(context0, dict):
            continue
        filename = context0.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            continue
        rows.append({"query": question, "expected_pdf": _normalize_pdf_name(filename)})

    if not rows:
        raise ValueError(f"financebench_json adapter found no valid rows in {query_json}")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv


_ADAPTER_HANDLERS: dict[str, tuple[Callable[[Path, Path], Path], str]] = {
    "page_plus_one": (_adapt_page_plus_one, "query_adapter.page_plus_one.csv"),
    "financebench_json": (_adapt_financebench_json, "query_adapter.financebench_json.csv"),
}


def prepare_recall_query_file(*, query_csv: Path | None, recall_adapter: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if query_csv is None:
        return output_dir / "__query_csv_missing__.csv"

    adapter = str(recall_adapter or "none").strip().lower()
    if adapter not in VALID_RECALL_ADAPTERS:
        raise ValueError(f"Unknown recall adapter '{recall_adapter}'. Valid adapters: {sorted(VALID_RECALL_ADAPTERS)}")

    source = Path(query_csv)
    if adapter == "none":
        return source

    handler = _ADAPTER_HANDLERS.get(adapter)
    if handler is None:
        raise ValueError(f"Adapter '{adapter}' is valid but not implemented.")

    adapter_fn, output_name = handler
    return adapter_fn(source, output_dir / output_name)
