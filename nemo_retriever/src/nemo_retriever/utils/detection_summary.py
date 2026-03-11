# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared detection summary logic.

Provides a single function that accumulates per-page detection counters from
an iterable of ``(page_key, metadata_dict, row_dict)`` tuples.  Both the
batch pipeline (reading from LanceDB) and inprocess pipeline (reading from
a DataFrame) can produce these tuples, allowing the summary computation to
be shared.
"""

from __future__ import annotations

from datetime import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def compute_detection_summary(
    rows: Iterable[Tuple[Any, Dict[str, Any], Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute deduped detection totals from an iterable of page data.

    Each element is ``(page_key, metadata_dict, row_dict)`` where:

    - *page_key* is a hashable value used to deduplicate exploded content rows
      (e.g. ``(source_id, page_number)``).
    - *metadata_dict* is the parsed JSON metadata (may contain counters from the
      LanceDB metadata column or from direct DataFrame columns).
    - *row_dict* is the raw row dict, used as fallback for counters stored as
      top-level DataFrame columns (e.g. ``table``, ``chart`` lists).
    """
    per_page: dict[Any, dict] = {}

    for page_key, meta, raw_row in rows:
        entry = per_page.setdefault(
            page_key,
            {
                "pe": 0,
                "ocr_table": 0,
                "ocr_chart": 0,
                "ocr_infographic": 0,
                "pe_by_label": defaultdict(int),
            },
        )

        pe = _safe_int(meta.get("page_elements_v3_num_detections") or raw_row.get("page_elements_v3_num_detections"))
        entry["pe"] = max(entry["pe"], pe)

        for field, meta_key, col_key in [
            ("ocr_table", "ocr_table_detections", "table"),
            ("ocr_chart", "ocr_chart_detections", "chart"),
            ("ocr_infographic", "ocr_infographic_detections", "infographic"),
        ]:
            val = _safe_int(meta.get(meta_key))
            if val == 0:
                col_val = raw_row.get(col_key)
                if isinstance(col_val, list):
                    val = len(col_val)
            entry[field] = max(entry[field], val)

        label_counts = meta.get("page_elements_v3_counts_by_label") or raw_row.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                entry["pe_by_label"][str(label)] = max(
                    entry["pe_by_label"][str(label)],
                    _safe_int(count),
                )

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    pe_total = ocr_table_total = ocr_chart_total = ocr_infographic_total = 0
    for e in per_page.values():
        pe_total += e["pe"]
        ocr_table_total += e["ocr_table"]
        ocr_chart_total += e["ocr_chart"]
        ocr_infographic_total += e["ocr_infographic"]
        for label, count in e["pe_by_label"].items():
            pe_by_label_totals[label] += count

    return {
        "pages_seen": len(per_page),
        "page_elements_v3_total_detections": pe_total,
        "page_elements_v3_counts_by_label": dict(sorted(pe_by_label_totals.items())),
        "ocr_table_total_detections": ocr_table_total,
        "ocr_chart_total_detections": ocr_chart_total,
        "ocr_infographic_total_detections": ocr_infographic_total,
    }


def iter_lancedb_rows(uri: str, table_name: str):
    """Yield ``(page_key, meta, row_dict)`` tuples from a LanceDB table."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    df = table.to_pandas()[["source_id", "page_number", "metadata"]]

    for row in df.itertuples(index=False):
        source_id = str(getattr(row, "source_id", "") or "")
        page_number = _safe_int(getattr(row, "page_number", -1), default=-1)
        raw_metadata = getattr(row, "metadata", None)
        meta: dict = {}
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                pass
        yield (source_id, page_number), meta, {}


def iter_dataframe_rows(df):
    """Yield ``(page_key, meta, row_dict)`` tuples from a pandas DataFrame."""
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        path = str(row_dict.get("path") or row_dict.get("source_id") or "")
        page_number = _safe_int(row_dict.get("page_number", -1), default=-1)

        meta = row_dict.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        yield (path, page_number), meta, row_dict


def collect_detection_summary_from_lancedb(uri: str, table_name: str) -> Optional[Dict[str, Any]]:
    """Collect detection summary from a LanceDB table."""
    try:
        return compute_detection_summary(iter_lancedb_rows(uri, table_name))
    except Exception:
        return None


def collect_detection_summary_from_df(df) -> Dict[str, Any]:
    """Collect detection summary from a pandas DataFrame."""
    return compute_detection_summary(iter_dataframe_rows(df))


def print_detection_summary(summary: Optional[Dict[str, Any]]) -> None:
    """Print a detection summary to stdout."""
    if summary is None:
        print("Detection summary: unavailable (could not read metadata).")
        return
    print("\nDetection summary (deduped by source_id/page_number):")
    print(f"  Pages seen: {summary['pages_seen']}")
    print(f"  PageElements v3 total detections: {summary['page_elements_v3_total_detections']}")
    print(f"  OCR table detections: {summary['ocr_table_total_detections']}")
    print(f"  OCR chart detections: {summary['ocr_chart_total_detections']}")
    print(f"  OCR infographic detections: {summary['ocr_infographic_total_detections']}")
    print("  PageElements v3 counts by label:")
    by_label = summary.get("page_elements_v3_counts_by_label") or {}
    if not by_label:
        print("    (none)")
    else:
        for label, count in by_label.items():
            print(f"    {label}: {count}")


def write_detection_summary(path: Path, summary: Optional[Dict[str, Any]]) -> None:
    """Write a detection summary dict to a JSON file."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = summary if summary is not None else {"error": "Detection summary unavailable."}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    """Print pages-per-second throughput to stdout."""
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print("Pages/sec: unavailable (could not estimate processed pages). " f"Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only; excludes Ray startup and recall): {pps:.2f}")


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``raw / H:MM:SS.mmm``."""
    ms = int(round(seconds * 1000))
    h, remainder = divmod(ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, millis = divmod(remainder, 1000)
    return f"{seconds:.2f}s / {h}:{m:02d}:{s:02d}.{millis:03d}"


def print_run_summary(
    processed_pages: Optional[int],
    input_path: Path,
    hybrid: bool,
    lancedb_uri: str,
    lancedb_table_name: str,
    total_time: float,
    ingest_only_total_time: float,
    ray_dataset_download_total_time: float,
    lancedb_write_total_time: float,
    recall_total_time: float,
    recall_metrics: Dict[str, float],
) -> None:
    ingest_only_pps = processed_pages / ingest_only_total_time
    ingest_and_lancedb_write_pps = processed_pages / (ingest_only_total_time + lancedb_write_total_time)
    recall_qps = processed_pages / recall_total_time
    total_pps = processed_pages / total_time
    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    print(f"===== Run Summary - {utc_now} UTC =====")

    print("Run Configuration:")
    print(f"\tInput path: {input_path}")
    print(f"\tHybrid: {hybrid}")
    print(f"\tLancedb URI: {lancedb_uri}")
    print(f"\tLancedb Table: {lancedb_table_name}")

    print("Runtimes:")
    print(f"\tTotal pages processed: {processed_pages} from {input_path}")
    print(f"\tIngestion only time: {_fmt_time(ingest_only_total_time)}")
    print(f"\tRay dataset download time: {_fmt_time(ray_dataset_download_total_time)}")
    print(f"\tLanceDB Write Time: {_fmt_time(lancedb_write_total_time)}")
    print(f"\tRecall time: {_fmt_time(recall_total_time)}")

    print("PPS:")
    print(f"\tIngestion only PPS: {ingest_only_pps:.2f}")
    print(f"\tIngestion + LanceDB Write PPS: {ingest_and_lancedb_write_pps:.2f}")
    print(f"\tRecall QPS: {recall_qps:.2f}")
    print(f"\tTotal - Processed: {processed_pages} pages in {_fmt_time(total_time)} @ {total_pps:.2f} PPS")

    print("Recall metrics:")
    for k, v in recall_metrics.items():
        print(f"  {k}: {v:.4f}")
