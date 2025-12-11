#!/usr/bin/env python3
"""
Visualize per-stage resident times from scripts/tests artifacts.

Usage:
    python scripts/tests/tools/plot_stage_totals.py \
        scripts/tests/artifacts/<run>/results.json \
        --pipeline config/default_pipeline.yaml \
        --top-n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml

# Ensure repo root is importable so we can reuse shared trace helpers
CURRENT_DIR = Path(__file__).resolve()
try:
    REPO_ROOT = CURRENT_DIR.parents[3]
except IndexError:
    REPO_ROOT = CURRENT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.tests.utils.trace_loader import ensure_trace_summary
except ImportError as err:  # pragma: no cover - defensive fallback
    ensure_trace_summary = None  # type: ignore[assignment]
    print(f"Warning: failed to import trace_loader helpers ({err}); trace summary rebuild disabled.")


FRIENDLY_STAGE_NAMES: Dict[str, str] = {
    "source_stage": "Source",
    "metadata_injector": "Metadata Injector",
    "metadata_injector_channel_in": "Metadata Queue",
    "pdf_extractor": "PDF Extractor",
    "pdf_extractor_channel_in": "PDF Queue",
    "audio_extractor": "Audio Extractor",
    "audio_extractor_channel_in": "Audio Queue",
    "docx_extractor": "DOCX Extractor",
    "docx_extractor_channel_in": "DOCX Queue",
    "pptx_extractor": "PPTX Extractor",
    "pptx_extractor_channel_in": "PPTX Queue",
    "image_extractor": "Image Extractor",
    "image_extractor_channel_in": "Image Queue",
    "html_extractor": "HTML Extractor",
    "html_extractor_channel_in": "HTML Queue",
    "infographic_extractor": "Infographic Extractor",
    "infographic_extractor_channel_in": "Infographic Queue",
    "table_extractor": "Table Extractor",
    "table_extractor_channel_in": "Table Queue",
    "chart_extractor": "Chart Extractor",
    "chart_extractor_channel_in": "Chart Queue",
    "image_filter": "Image Filter",
    "image_filter_channel_in": "Image Filter Queue",
    "image_dedup": "Image Dedup",
    "image_dedup_channel_in": "Image Dedup Queue",
    "text_splitter": "Text Splitter",
    "text_splitter_channel_in": "Text Splitter Queue",
    "image_caption": "Image Caption",
    "image_caption_channel_in": "Image Caption Queue",
    "text_embedder": "Text Embedder",
    "text_embedder_channel_in": "Text Embed Queue",
    "image_storage": "Image Storage",
    "image_storage_channel_in": "Image Storage Queue",
    "embedding_storage": "Embedding Storage",
    "embedding_storage_channel_in": "Embedding Storage Queue",
    "broker_response": "Broker Response",
    "broker_source_network_in": "Broker Network",
    "message_broker_task_source": "Message Broker Source",
}
PDFIUM_STAGE_SUFFIXES = ("render_page", "bitmap_to_numpy", "scale_image", "pad_image")
PDFIUM_STAGE_NAMES = {f"pdf_extractor::pdf_extraction::{suffix}": suffix for suffix in PDFIUM_STAGE_SUFFIXES}
PDFIUM_STAGE_ORDER = [f"pdf_extractor::pdf_extraction::{suffix}" for suffix in PDFIUM_STAGE_SUFFIXES]
PDFIUM_STAGE_COLORS = {
    PDFIUM_STAGE_ORDER[0]: "#4C72B0",  # render_page
    PDFIUM_STAGE_ORDER[1]: "#55A868",  # bitmap_to_numpy
    PDFIUM_STAGE_ORDER[2]: "#C44E52",  # scale_image
    PDFIUM_STAGE_ORDER[3]: "#8172B2",  # pad_image
}


def load_stage_order(pipeline_yaml: Path) -> List[str]:
    pipeline = yaml.safe_load(pipeline_yaml.read_text())
    return [stage["name"] for stage in pipeline.get("stages", [])]


def friendly_name(stage: str) -> str:
    if stage in PDFIUM_STAGE_NAMES:
        suffix = PDFIUM_STAGE_NAMES[stage].replace("_", " ").title()
        return f"{FRIENDLY_STAGE_NAMES['pdf_extractor']} :: {suffix}"
    if stage in FRIENDLY_STAGE_NAMES:
        return FRIENDLY_STAGE_NAMES[stage]
    if stage.endswith("_channel_in"):
        base = stage.removesuffix("_channel_in")
        return f"{FRIENDLY_STAGE_NAMES.get(base, base)} Queue"
    # Collapse nested trace stages
    if "::" in stage:
        parts = stage.split("::")
        if parts[0] in FRIENDLY_STAGE_NAMES:
            tail = parts[2:] if len(parts) > 2 else parts[1:]
            if not tail:
                tail = parts[1:]
            return f"{FRIENDLY_STAGE_NAMES[parts[0]]} :: {'::'.join(tail)}"
    return stage


def stage_sort_key(stage: str, order_map: Dict[str, int]) -> Tuple[int, str]:
    base = stage.split("::")[0]
    base = base.replace("_channel_in", "")
    return order_map.get(base, len(order_map)), stage


def should_keep_stage(stage: str, keep_nested: bool, exclude_network: bool) -> bool:
    if exclude_network and ("broker_source_network_in" in stage or "network_in" in stage):
        return False
    if "::pdfium_pages_to_numpy" in stage:
        return False
    if keep_nested:
        return True
    return "::" not in stage


def build_stage_plot(
    data: Dict[str, Any],
    results_path: Path,
    pipeline_yaml: Path,
    top_n: int | None,
    log_scale: bool,
    width: int,
    keep_nested: bool,
    sort_mode: str,
    summary_rows: int,
    exclude_network: bool,
    stage_metric: str,
):
    trace_summary = data.get("trace_summary")
    if not trace_summary:
        print("No trace_summary present; skipping stage plot.")
        return

    base_stage_totals = trace_summary.get("stage_totals", {})
    nested_stage_totals = trace_summary.get("nested_stage_totals", {})
    if not base_stage_totals and not nested_stage_totals:
        print("No stage_totals found in trace_summary; skipping stage plot.")
        return

    stage_totals = dict(base_stage_totals)

    stage_order = load_stage_order(pipeline_yaml)
    order_map = {stage: idx for idx, stage in enumerate(stage_order)}

    merged: Dict[str, Dict[str, float]] = {}

    def metric_value(stats: Dict[str, float]) -> float:
        if stage_metric == "average":
            return stats.get("avg_resident_s", 0.0)
        return stats.get("total_resident_s", 0.0)

    def metric_label() -> str:
        return "Average resident seconds per document" if stage_metric == "average" else "Total resident seconds"

    for stage, stats in stage_totals.items():
        if not should_keep_stage(stage, keep_nested=keep_nested, exclude_network=exclude_network):
            continue
        merged[stage] = stats

    entries = []
    for stage, stats in merged.items():
        sort_key = stage_sort_key(stage, order_map)
        entries.append((sort_key, stage, stats))

    if sort_mode == "total":
        entries.sort(key=lambda item: metric_value(item[2]), reverse=True)
    else:
        entries.sort()

    if top_n:
        entries = entries[:top_n]

    labeled_entries = []
    for _, stage, stats in entries:
        labeled_entries.append(
            {
                "raw_stage": stage,
                "label": friendly_name(stage),
                "value": metric_value(stats),
                "total": stats.get("total_resident_s", 0.0),
                "avg": stats.get("avg_resident_s", 0.0),
                "docs": stats.get("documents", 0),
            }
        )

    stages = [item["label"] for item in labeled_entries]
    totals = [item["value"] for item in labeled_entries]

    fig_height = max(6, len(stages) * 0.35)
    fig, ax = plt.subplots(figsize=(width, fig_height))
    bars = ax.barh(range(len(stages) - 1, -1, -1), totals, color="#4C72B0")
    ax.set_xlabel(metric_label())
    ax.set_yticks(range(len(stages) - 1, -1, -1))
    ax.set_yticklabels(stages)
    if log_scale:
        ax.set_xscale("log")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    def _format_seconds(value: float) -> str:
        if value == 0:
            return "0s"
        if value < 0.1:
            return f"{value:.3f}s"
        if value < 10:
            return f"{value:.2f}s"
        return f"{value:.1f}s"

    for bar, entry in zip(bars, labeled_entries):
        width_val = bar.get_width()
        ax.text(
            width_val,
            bar.get_y() + bar.get_height() / 2,
            f"{_format_seconds(width_val)} (avg {entry['avg']:.2f}s, docs {entry['docs']})",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    title_prefix = "Average resident time per document" if stage_metric == "average" else "Stage resident time totals"
    ax.set_title(f"{title_prefix} – {data['test_config']['test_name']}")
    plt.tight_layout()

    out_path = results_path.with_suffix(".stage_time.png")
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    if summary_rows > 0:
        print("\nTop stages by resident time")
        print("-" * 90)
        print(f"{'Stage':<40} {'Total (s)':>12} {'Avg (s)':>10} {'Docs':>6}")
        print("-" * 90)
        for item in labeled_entries[:summary_rows]:
            stage = item["label"]
            total = item["total"]
            avg = item["avg"]
            doc_cnt = item["docs"]
            print(f"{stage:<40} {total:>12.2f} {avg:>10.2f} {doc_cnt:>6}")

    _print_pdfium_breakdown(trace_summary)


def _doc_wait_seconds(doc: Dict[str, Any]) -> float | None:
    wait = doc.get("ray_wait_s")
    if wait is not None:
        try:
            return float(wait)
        except (TypeError, ValueError):
            return None
    start = doc.get("ray_start_ts_s")
    submitted = doc.get("submission_ts_s")
    if start is None or submitted is None:
        return None
    try:
        wait_val = float(start) - float(submitted)
    except (TypeError, ValueError):
        return None
    return max(0.0, wait_val)


def _percentile(values: List[float], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] + (values[c] - values[f]) * (k - f)


def _print_wait_summary(wait_values: List[float], wall_values: List[float]):
    if not wait_values or not wall_values:
        return
    sorted_waits = sorted(wait_values)
    p50 = _percentile(sorted_waits, 50)
    p90 = _percentile(sorted_waits, 90)
    p99 = _percentile(sorted_waits, 99)
    max_wait = sorted_waits[-1]
    ratio_samples = [w / w_tot for w, w_tot in zip(wait_values, wall_values) if w_tot > 0]
    avg_ratio = sum(ratio_samples) / len(ratio_samples) if ratio_samples else 0.0
    print("\nWait time summary (all documents)")
    print("-" * 90)
    print(f"Median wait: {p50:.2f}s | p90: {p90:.2f}s | p99: {p99:.2f}s | max: {max_wait:.2f}s")
    print(f"Average wait / wall fraction: {avg_ratio * 100:.2f}%")


def _print_queue_summary(queue_values: List[float], wall_values: List[float]):
    if not queue_values or not wall_values:
        return
    sorted_queues = sorted(queue_values)
    p50 = _percentile(sorted_queues, 50)
    p90 = _percentile(sorted_queues, 90)
    p99 = _percentile(sorted_queues, 99)
    max_queue = sorted_queues[-1]
    ratios = [queue / wall for queue, wall in zip(queue_values, wall_values) if wall > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    print("\nIn-Ray queue time summary (all documents)")
    print("-" * 90)
    print(f"Median queue: {p50:.2f}s | p90: {p90:.2f}s | p99: {p99:.2f}s | max: {max_queue:.2f}s")
    print(f"Average queue / wall fraction: {avg_ratio * 100:.2f}%")


def _pdfium_doc_entries(trace_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_breakdown = trace_summary.get("pdfium_document_breakdown")
    if not doc_breakdown:
        return []

    aggregated: Dict[str, Dict[str, Any]] = {}
    for doc in doc_breakdown:
        source_id = doc.get("source_id") or f"document_{doc.get('document_index')}"
        stage_seconds = doc.get("stage_seconds") or {}
        if not stage_seconds:
            continue
        agg = aggregated.setdefault(
            source_id,
            {
                "document_indices": [],
                "source_id": source_id,
                "stage_seconds": defaultdict(float),
                "page_count": 0,
            },
        )
        agg["document_indices"].append(doc.get("document_index"))
        agg["page_count"] += int(doc.get("page_count") or 0)
        for stage, value in stage_seconds.items():
            agg["stage_seconds"][stage] += float(value)

    entries = []
    for agg in aggregated.values():
        stage_seconds = dict(agg["stage_seconds"])
        if not stage_seconds:
            continue
        total_seconds = sum(stage_seconds.values())
        page_count = agg["page_count"]
        per_page = total_seconds / page_count if page_count else None
        doc_indices = [idx for idx in agg["document_indices"] if idx is not None]
        representative_idx = min(doc_indices) if doc_indices else None
        entries.append(
            {
                "document_index": representative_idx,
                "source_id": agg["source_id"],
                "page_count": page_count,
                "stage_seconds": stage_seconds,
                "total_seconds": total_seconds,
                "per_page_seconds": per_page,
            }
        )

    entries.sort(key=lambda item: (item["document_index"] if item["document_index"] is not None else math.inf))
    return entries


def build_pdfium_doc_plot(
    trace_summary: Dict[str, Any],
    results_path: Path,
    width: int,
    top_n: int,
    metric: str,
):
    entries = _pdfium_doc_entries(trace_summary)
    if not entries:
        print("No pdfium per-document breakdown found; skipping PDFium stacked plot.")
        return

    if metric == "per_page":
        entries = [entry for entry in entries if entry["page_count"]]

        def key_func(entry):
            return entry["per_page_seconds"] or 0.0

        value_label = "Avg seconds per page"
    else:

        def key_func(entry):
            return entry["total_seconds"]

        value_label = "Total seconds per document"

    entries.sort(key=key_func, reverse=True)
    if top_n:
        entries = entries[:top_n]
    if not entries:
        print("PDFium doc plot: no entries after filtering; skipping.")
        return

    labels = []
    for entry in entries:
        source_id = entry.get("source_id") or f"document_{entry['document_index']}"
        source_name = Path(source_id).name
        doc_idx = entry.get("document_index")
        label = f"{doc_idx}: {source_name}" if doc_idx is not None else source_name
        page_count = entry.get("page_count", 0)
        if page_count:
            label = f"{label} ({page_count}p)"
        labels.append(label)

    fig_height = max(6, len(entries) * 0.4)
    fig, ax = plt.subplots(figsize=(width, fig_height))
    positions = list(range(len(entries)))
    stacked = [0.0 for _ in entries]
    legend_added: Dict[str, bool] = {}

    for stage in PDFIUM_STAGE_ORDER:
        stage_values = []
        for idx, entry in enumerate(entries):
            stage_seconds = entry["stage_seconds"].get(stage, 0.0)
            if metric == "per_page":
                page_count = entry["page_count"]
                stage_values.append(stage_seconds / page_count if page_count else 0.0)
            else:
                stage_values.append(stage_seconds)
        if not any(stage_values):
            continue
        ax.barh(
            positions,
            stage_values,
            left=stacked,
            color=PDFIUM_STAGE_COLORS.get(stage, "#999999"),
            label=friendly_name(stage) if not legend_added.get(stage) else None,
            height=0.5,
        )
        legend_added[stage] = True
        stacked = [prev + cur for prev, cur in zip(stacked, stage_values)]

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(value_label)
    ax.set_title("PDFium per-document breakdown")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = results_path.with_suffix(".pdfium_docs.png")
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def export_pdfium_csv(trace_summary: Dict[str, Any], results_path: Path):
    entries = _pdfium_doc_entries(trace_summary)
    if not entries:
        print("No pdfium per-document breakdown found; skipping CSV export.")
        return

    csv_path = results_path.with_suffix(".pdfium_breakdown.csv")
    with csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["document_index", "source_id", "page_count", "metric", "seconds", "seconds_per_page"])
        for entry in entries:
            page_count = entry.get("page_count", 0)
            for stage in PDFIUM_STAGE_ORDER:
                seconds = entry["stage_seconds"].get(stage)
                if seconds is None:
                    continue
                seconds_per_page = seconds / page_count if page_count else ""
                writer.writerow(
                    [
                        entry.get("document_index"),
                        entry.get("source_id"),
                        page_count,
                        PDFIUM_STAGE_NAMES.get(stage, stage),
                        f"{seconds:.6f}",
                        f"{seconds_per_page:.6f}" if seconds_per_page != "" else "",
                    ]
                )
    print(f"Wrote {csv_path}")


def _print_pdfium_breakdown(trace_summary: Dict[str, Any]):
    samples = trace_summary.get("nested_stage_samples") or {}
    if not samples:
        return
    nested_totals = trace_summary.get("nested_stage_totals", {})
    entries = []
    for stage in PDFIUM_STAGE_NAMES:
        values = samples.get(stage)
        if not values:
            continue
        sorted_vals = sorted(values)
        median = _percentile(sorted_vals, 50) or 0.0
        p90 = _percentile(sorted_vals, 90) or 0.0
        max_val = sorted_vals[-1] if sorted_vals else 0.0
        stats = nested_totals.get(stage, {})
        total = stats.get("total_resident_s", sum(values))
        avg = stats.get("avg_resident_s", total / len(values) if values else 0.0)
        docs = stats.get("documents", len(values))
        label = friendly_name(stage)
        entries.append(
            {
                "label": label,
                "total": total,
                "avg": avg,
                "median": median,
                "p90": p90,
                "max": max_val,
                "docs": docs,
            }
        )

    if not entries:
        return

    print("\nPDFium extraction breakdown (per stage)")
    print("-" * 100)
    print(f"{'Stage':<40} {'Total (s)':>12} {'Avg (s)':>10} {'Median':>10} {'P90':>10} {'Max':>10} {'Docs':>6}")
    print("-" * 100)
    for item in entries:
        print(
            f"{item['label']:<40} {item['total']:>12.2f} {item['avg']:>10.2f} "
            f"{item['median']:>10.2f} {item['p90']:>10.2f} {item['max']:>10.2f} {item['docs']:>6}"
        )


def build_wall_plot(
    data: Dict[str, Any],
    results_path: Path,
    doc_top_n: int | None,
    width: int,
    summary_rows: int,
    title_suffix: str | None = None,
    doc_sort: str = "wall",
):
    trace_summary = data.get("trace_summary")
    if not trace_summary:
        print("No trace_summary present; skipping wall-time plot.")
        return
    document_totals = trace_summary.get("document_totals")
    if not document_totals:
        print("No document_totals present; skipping wall-time plot.")
        return

    run_results = data.get("results", {})
    ingestion_time = run_results.get("ingestion_time_s")
    result_count = run_results.get("result_count") or len(document_totals)

    def wait_sort_key(doc: Dict[str, Any]) -> float:
        wait_val = _doc_wait_seconds(doc)
        return wait_val if wait_val is not None else 0.0

    if doc_sort == "wait":
        documents = sorted(document_totals, key=wait_sort_key, reverse=True)
    else:
        documents = sorted(
            document_totals,
            key=lambda item: item.get("total_wall_s", 0.0),
            reverse=True,
        )
    if doc_top_n:
        documents = documents[:doc_top_n]

    if not documents:
        print("No document entries available after filtering; skipping wall-time plot.")
        return

    labels: List[str] = []
    wall_vals: List[float] = []
    resident_vals: List[float] = []
    ratios: List[float] = []
    ray_start_vals: List[float | None] = []
    ray_end_vals: List[float | None] = []
    submission_vals: List[float | None] = []
    wait_vals: List[float | None] = []
    queue_vals: List[float | None] = []

    for doc in documents:
        source = doc.get("source_id") or f"doc_{doc.get('document_index', '?')}"
        source_name = Path(source).name
        doc_idx = doc.get("document_index")
        label = f"{doc_idx}: {source_name}" if doc_idx is not None else source_name
        labels.append(label)
        wall = float(doc.get("total_wall_s", 0.0))
        resident = float(doc.get("total_resident_s", 0.0))
        wall_vals.append(wall)
        resident_vals.append(resident)
        ratios.append(resident / wall if wall > 0 else math.inf)
        ray_start_vals.append(doc.get("ray_start_ts_s"))
        ray_end_vals.append(doc.get("ray_end_ts_s"))
        submission_vals.append(doc.get("submission_ts_s"))
        wait_vals.append(_doc_wait_seconds(doc))
        queue_val = doc.get("in_ray_queue_s")
        if queue_val is None:
            queue_vals.append(None)
        else:
            try:
                queue_vals.append(float(queue_val))
            except (TypeError, ValueError):
                queue_vals.append(None)

    labels = labels[::-1]
    wall_vals = wall_vals[::-1]
    resident_vals = resident_vals[::-1]
    ratios = ratios[::-1]
    wait_vals = wait_vals[::-1]
    ray_start_vals = ray_start_vals[::-1]
    ray_end_vals = ray_end_vals[::-1]
    submission_vals = submission_vals[::-1]
    queue_vals = queue_vals[::-1]

    base_positions = list(range(len(labels)))
    wall_positions = [pos + 0.2 for pos in base_positions]
    resident_positions = [pos - 0.2 for pos in base_positions]
    queue_positions = [pos + 0.05 for pos in base_positions]

    fig_height = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(width, fig_height))

    ax.barh(
        wall_positions,
        wall_vals,
        height=0.35,
        color="#55A868",
        label="Wall seconds",
    )
    ax.barh(
        resident_positions,
        resident_vals,
        height=0.35,
        color="#C44E52",
        label="Resident seconds",
    )
    wait_display_vals = [val if val is not None else 0.0 for val in wait_vals]
    ax.barh(
        base_positions,
        wait_display_vals,
        height=0.15,
        color="#FFA600",
        label="Wait before Ray",
    )
    queue_display_vals = [val if val is not None else 0.0 for val in queue_vals]
    ax.barh(
        queue_positions,
        queue_display_vals,
        height=0.12,
        color="#AA65D2",
        label="In-Ray queue",
    )

    ax.set_yticks(base_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds per document")
    title = "Document wall vs resident time"
    if title_suffix:
        title = f"{title} – {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="upper right")

    max_val = max(wall_vals + resident_vals)
    text_x = max_val * 1.02 if max_val > 0 else 0.5

    for pos, wall, resident, ratio in zip(base_positions, wall_vals, resident_vals, ratios):
        if wall <= 0 and resident <= 0:
            continue
        ratio_str = "∞" if math.isinf(ratio) else f"{ratio:.1f}×"
        ax.text(
            text_x,
            pos,
            f"resident/wall {ratio_str}",
            va="center",
            fontsize=8,
            color="#333333",
        )

    info_lines = []
    if ingestion_time is not None:
        info_lines.append(f"Run wall time: {ingestion_time:.1f}s")
    if result_count:
        info_lines.append(f"Documents: {result_count}")
    if info_lines:
        ax.text(
            0.01,
            1.02,
            " • ".join(info_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()
    out_path = results_path.with_suffix(".wall_time.png")
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")

    all_wait_values: List[float] = []
    all_queue_values: List[float] = []
    all_walls: List[float] = []
    for doc in document_totals:
        wait_value = _doc_wait_seconds(doc)
        if wait_value is None:
            continue
        all_wait_values.append(wait_value)
        wall_value = doc.get("total_wall_s")
        if wall_value is None:
            all_walls.append(wait_value)  # Dummy placeholder to keep lengths equal
        else:
            try:
                all_walls.append(float(wall_value))
            except (TypeError, ValueError):
                all_walls.append(wait_value)
        queue_val = doc.get("in_ray_queue_s")
        if queue_val is not None:
            try:
                all_queue_values.append(float(queue_val))
            except (TypeError, ValueError):
                pass
    _print_wait_summary(all_wait_values, all_walls)
    _print_queue_summary(all_queue_values, all_walls)

    has_ray = any(value is not None for value in ray_start_vals)
    has_submission = any(value is not None for value in submission_vals)
    has_wait = any(value is not None for value in wait_vals)
    has_queue = any(value is not None for value in queue_vals)

    if summary_rows > 0:
        print("\nTop documents by wall time")
        print("-" * 90)
        header = f"{'Document':<40} {'Wall (s)':>12} {'Resident (s)':>14} {'Ratio':>8}"
        if has_wait:
            header += f" {'Wait (s)':>10}"
        if has_queue:
            header += f" {'Ray queue (s)':>14}"
        if has_submission:
            header += f" {'Submitted (s)':>15}"
        if has_ray:
            header += f" {'Ray start (s)':>15} {'Ray end (s)':>15}"
        print(header)
        print("-" * 90)
        truncated = list(
            zip(
                labels[::-1],
                wall_vals[::-1],
                resident_vals[::-1],
                ratios[::-1],
                wait_vals[::-1],
                queue_vals[::-1],
                submission_vals[::-1],
                ray_start_vals[::-1],
                ray_end_vals[::-1],
            )
        )[:summary_rows]
        for label, wall, resident, ratio, wait, queue, submitted, ray_start, ray_end in truncated:
            ratio_str = "∞" if math.isinf(ratio) else f"{ratio:.2f}"
            row = f"{label:<40} {wall:>12.2f} {resident:>14.2f} {ratio_str:>8}"
            if has_wait:
                row += f" {wait if wait is not None else '-':>10}"
            if has_queue:
                row += f" {queue if queue is not None else '-':>10}"
            if has_submission:
                row += f" {submitted if submitted is not None else '-':>15}"
            if has_ray:
                row += f" {ray_start if ray_start is not None else '-':>15}"
                row += f" {ray_end if ray_end is not None else '-':>15}"
            print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage resident times from results.json")
    parser.add_argument("results", type=Path, help="Path to results.json artifact")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Optional directory containing raw trace *.json files (auto-detected if omitted)",
    )
    parser.add_argument(
        "--pipeline",
        type=Path,
        default=Path("config/default_pipeline.yaml"),
        help="Pipeline YAML to derive stage ordering",
    )
    parser.add_argument("--top-n", type=int, help="Limit to top N stages")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale on x-axis")
    parser.add_argument("--width", type=int, default=14, help="Plot width in inches")
    parser.add_argument(
        "--keep-nested",
        action="store_true",
        help="Keep nested stage names (default: drop entries containing '::')",
    )
    parser.add_argument(
        "--sort",
        choices=["total", "pipeline"],
        default="total",
        help="Sort bars by total resident seconds or pipeline order",
    )
    parser.add_argument(
        "--stage-metric",
        choices=["total", "average"],
        default="total",
        help="Bar length metric for stage plot (total seconds or avg per document)",
    )
    parser.add_argument(
        "--summary-rows",
        type=int,
        default=10,
        help="Print textual summary for top N stages (0 disables)",
    )
    parser.add_argument(
        "--exclude-network",
        action="store_true",
        help="Exclude broker/network-in stages from the visualization",
    )
    parser.add_argument(
        "--doc-top-n",
        type=int,
        default=30,
        help="Limit wall-time plot to top N documents by wall seconds (0 disables limit)",
    )
    parser.add_argument(
        "--doc-summary-rows",
        type=int,
        default=10,
        help="Print textual summary for document wall times (0 disables)",
    )
    parser.add_argument(
        "--doc-sort",
        choices=["wall", "wait"],
        default="wall",
        help="Sort document chart by wall time or wait time",
    )
    parser.add_argument(
        "--skip-wall-plot",
        action="store_true",
        help="Only emit the stage resident-time plot",
    )
    parser.add_argument(
        "--skip-pdfium-doc-plot",
        action="store_true",
        help="Disable the PDFium per-document stacked chart",
    )
    parser.add_argument(
        "--pdfium-doc-top-n",
        type=int,
        default=15,
        help="Show top N documents in PDFium stacked plot (0 shows all)",
    )
    parser.add_argument(
        "--pdfium-doc-metric",
        choices=["total", "per_page"],
        default="per_page",
        help="Sort/scale PDFium doc plot by total seconds or per-page average",
    )
    parser.add_argument(
        "--pdfium-export-csv",
        action="store_true",
        help="Export per-document PDFium breakdown to CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.results.read_text())
    if ensure_trace_summary is None:
        print("trace_loader helpers unavailable; proceeding with existing trace_summary.")
    else:
        ensure_trace_summary(data, args.results, args.trace_dir)
    build_stage_plot(
        data=data,
        results_path=args.results,
        pipeline_yaml=args.pipeline,
        top_n=args.top_n,
        log_scale=args.log_scale,
        width=args.width,
        keep_nested=args.keep_nested,
        sort_mode=args.sort,
        summary_rows=args.summary_rows,
        exclude_network=args.exclude_network,
        stage_metric=args.stage_metric,
    )
    trace_summary = data.get("trace_summary", {})
    if (not args.skip_pdfium_doc_plot) and trace_summary:
        build_pdfium_doc_plot(
            trace_summary=trace_summary,
            results_path=args.results,
            width=args.width,
            top_n=args.pdfium_doc_top_n,
            metric=args.pdfium_doc_metric,
        )
    if args.pdfium_export_csv and trace_summary:
        export_pdfium_csv(trace_summary, args.results)
    if not args.skip_wall_plot:
        test_name = data.get("test_config", {}).get("test_name")
        build_wall_plot(
            data=data,
            results_path=args.results,
            doc_top_n=args.doc_top_n or None,
            width=args.width,
            summary_rows=args.doc_summary_rows,
            title_suffix=test_name,
            doc_sort=args.doc_sort,
        )


if __name__ == "__main__":
    main()
