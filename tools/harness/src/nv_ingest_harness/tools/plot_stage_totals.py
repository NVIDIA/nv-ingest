#!/usr/bin/env python3
"""
Visualize per-stage resident times from scripts/tests artifacts.

Generates a single stage resident time bar chart with consistent defaults for reproducibility.
Filtering flags available for debugging purposes.

Usage:
    # Default: stage plot with clean defaults
    python scripts/tests/tools/plot_stage_totals.py \
        scripts/tests/artifacts/<run>/results.json

    # Include additional stages for debugging
    python scripts/tests/tools/plot_stage_totals.py \
        scripts/tests/artifacts/<run>/results.json --include-nested --include-queues
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def _is_queue_stage(stage: str) -> bool:
    return stage.endswith("_channel_in")


def _is_pdfium_micro_stage(stage: str) -> bool:
    """
    PDFium micro-spans are excluded by default from the stage plot for cleaner visualization.
    Use --include-pdfium-micro to include them for debugging.
    """
    if "pdf_extractor::pdf_extraction::" not in stage:
        return False
    return any(
        suffix in stage
        for suffix in (
            "::pdfium_pages_to_numpy",
            "::render_page",
            "::bitmap_to_numpy",
            "::scale_image",
            "::pad_image",
        )
    )


def should_keep_stage(
    stage: str,
    include_nested: bool,
    include_network: bool,
    include_queues: bool,
    include_pdfium_micro: bool,
) -> bool:
    # Network stages: exclude by default unless explicitly included
    if (not include_network) and ("broker_source_network_in" in stage or "network_in" in stage):
        return False
    # Queue stages: exclude by default unless explicitly included
    if (not include_queues) and _is_queue_stage(stage):
        return False
    # PDFium micro-stages: exclude from main stage plot by default (they have their own plot)
    if (not include_pdfium_micro) and _is_pdfium_micro_stage(stage):
        return False
    # Nested stages: exclude by default unless explicitly included
    if include_nested:
        return True
    return "::" not in stage


def build_stage_plot(
    data: Dict[str, Any],
    results_path: Path,
    include_nested: bool,
    include_network: bool,
    include_queues: bool,
    include_pdfium_micro: bool,
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

    # Hardcoded defaults for reproducibility
    pipeline_yaml = Path("config/default_pipeline.yaml")
    stage_order = load_stage_order(pipeline_yaml)
    order_map = {stage: idx for idx, stage in enumerate(stage_order)}
    width = 14
    summary_rows = 10

    merged: Dict[str, Dict[str, float]] = {}

    for stage, stats in stage_totals.items():
        if not should_keep_stage(
            stage,
            include_nested=include_nested,
            include_network=include_network,
            include_queues=include_queues,
            include_pdfium_micro=include_pdfium_micro,
        ):
            continue
        merged[stage] = stats

    entries = []
    for stage, stats in merged.items():
        sort_key = stage_sort_key(stage, order_map)
        entries.append((sort_key, stage, stats))

    # Sort by total resident seconds (hardcoded)
    entries.sort(key=lambda item: item[2].get("total_resident_s", 0.0), reverse=True)

    labeled_entries = []
    for _, stage, stats in entries:
        labeled_entries.append(
            {
                "raw_stage": stage,
                "label": friendly_name(stage),
                "value": stats.get("total_resident_s", 0.0),
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
    ax.set_xlabel("Total resident seconds")
    ax.set_yticks(range(len(stages) - 1, -1, -1))
    ax.set_yticklabels(stages)
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

    ax.set_title(f"Stage resident time totals – {data['test_config']['test_name']}")
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


def _pdfium_doc_entries(trace_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_breakdown = trace_summary.get("pdfium_document_breakdown")
    if not doc_breakdown:
        return []

    from collections import defaultdict

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
    width: int = 14,
    top_n: int = 15,
    metric: str = "per_page",
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
    import csv

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stage resident times from results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: stage plot with clean defaults (excludes queues, network, nested, PDFium micro)
  python plot_stage_totals.py artifacts/run/results.json

  # Include additional stages for debugging
  python plot_stage_totals.py artifacts/run/results.json --include-nested --include-queues

  # Generate PDFium breakdown plot (for benchmarking PDFium rendering changes)
  python plot_stage_totals.py artifacts/run/results.json --pdfium-plot --pdfium-csv
        """,
    )
    parser.add_argument("results", type=Path, help="Path to results.json artifact")

    parser.add_argument(
        "--trace-dir",
        type=Path,
        help="Directory containing raw trace *.json files (auto-detected if omitted)",
    )

    # Filtering options (for debugging)
    parser.add_argument(
        "--include-nested",
        action="store_true",
        help="Include nested stage names (default: excluded)",
    )
    parser.add_argument(
        "--include-network",
        action="store_true",
        help="Include broker/network-in stages (default: excluded)",
    )
    parser.add_argument(
        "--include-queues",
        action="store_true",
        help="Include *_channel_in queue stages (default: excluded)",
    )
    parser.add_argument(
        "--include-pdfium-micro",
        action="store_true",
        help="Include PDFium micro-spans in stage plot (default: excluded)",
    )

    # PDFium plot options (optional, for benchmarking PDFium rendering changes)
    parser.add_argument(
        "--pdfium-plot",
        action="store_true",
        help="Generate PDFium per-document breakdown plot (default: disabled)",
    )
    parser.add_argument(
        "--pdfium-csv",
        action="store_true",
        help="Export PDFium breakdown to CSV (requires --pdfium-plot)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.results.read_text())

    # Always rebuild trace summary (hardcoded for reproducibility)
    if ensure_trace_summary is None:
        print("trace_loader helpers unavailable; proceeding with existing trace_summary.")
    else:
        ensure_trace_summary(data, args.results, args.trace_dir, force=True)

    build_stage_plot(
        data=data,
        results_path=args.results,
        include_nested=args.include_nested,
        include_network=args.include_network,
        include_queues=args.include_queues,
        include_pdfium_micro=args.include_pdfium_micro,
    )

    # Optional PDFium plot (for benchmarking PDFium rendering changes)
    trace_summary = data.get("trace_summary", {})
    if args.pdfium_plot and trace_summary:
        build_pdfium_doc_plot(
            trace_summary=trace_summary,
            results_path=args.results,
        )
        if args.pdfium_csv:
            export_pdfium_csv(trace_summary, args.results)


if __name__ == "__main__":
    main()
