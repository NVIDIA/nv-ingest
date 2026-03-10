# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by multiple example pipeline scripts."""

from __future__ import annotations

from typing import Optional


def estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    """Estimate pages processed by counting unique (source, page_number) pairs.

    Falls back to table row count if page-level fields are unavailable.
    """
    try:
        import lancedb  # type: ignore

        db = lancedb.connect(uri)
        table = db.open_table(table_name)
    except Exception:
        return None

    try:
        df = table.to_pandas()
        source_col = "source" if "source" in df.columns else "path" if "path" in df.columns else "source_id"
        if source_col not in df.columns or "page_number" not in df.columns:
            raise KeyError("Missing source/page_number columns")
        return int(
            df.dropna(subset=[source_col, "page_number"])
            .drop_duplicates(subset=[source_col, "page_number"])
            .shape[0]
        )
    except Exception:
        try:
            return int(table.count_rows())
        except Exception:
            return None


def print_pages_per_second(
    processed_pages: Optional[int],
    ingest_elapsed_s: float,
    *,
    label: str = "ingest only",
) -> None:
    """Print a throughput summary line."""
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print(f"Pages/sec: unavailable (could not estimate processed pages). Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec ({label}): {pps:.2f}")
