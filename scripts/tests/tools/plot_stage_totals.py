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
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import yaml


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


def load_stage_order(pipeline_yaml: Path) -> List[str]:
    pipeline = yaml.safe_load(pipeline_yaml.read_text())
    return [stage["name"] for stage in pipeline.get("stages", [])]


def friendly_name(stage: str) -> str:
    if stage in FRIENDLY_STAGE_NAMES:
        return FRIENDLY_STAGE_NAMES[stage]
    if stage.endswith("_channel_in"):
        base = stage.removesuffix("_channel_in")
        return f"{FRIENDLY_STAGE_NAMES.get(base, base)} Queue"
    # Collapse nested trace stages
    if "::" in stage:
        parts = stage.split("::")
        if parts[0] in FRIENDLY_STAGE_NAMES:
            return f"{FRIENDLY_STAGE_NAMES[parts[0]]} :: {'::'.join(parts[2:])}"
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


def build_plot(
    results_path: Path,
    pipeline_yaml: Path,
    top_n: int | None,
    log_scale: bool,
    width: int,
    keep_nested: bool,
    sort_mode: str,
    summary_rows: int,
    exclude_network: bool,
):
    data = json.loads(results_path.read_text())
    stage_totals = data["trace_summary"]["stage_totals"]

    stage_order = load_stage_order(pipeline_yaml)
    order_map = {stage: idx for idx, stage in enumerate(stage_order)}

    merged: Dict[str, Dict[str, float]] = {}
    for stage, stats in stage_totals.items():
        if not should_keep_stage(stage, keep_nested=keep_nested, exclude_network=exclude_network):
            continue
        merged[stage] = stats

    entries = []
    for stage, stats in merged.items():
        sort_key = stage_sort_key(stage, order_map)
        entries.append((sort_key, stage, stats))

    if sort_mode == "total":
        entries.sort(key=lambda item: item[2]["total_resident_s"], reverse=True)
    else:
        entries.sort()

    if top_n:
        entries = entries[:top_n]

    labeled_entries = [
        {
            "raw_stage": stage,
            "label": friendly_name(stage),
            "total": stats["total_resident_s"],
            "avg": stats["avg_resident_s"],
            "docs": stats["documents"],
        }
        for _, stage, stats in entries
    ]

    stages = [item["label"] for item in labeled_entries]
    totals = [item["total"] for item in labeled_entries]
    avgs = [item["avg"] for item in labeled_entries]
    docs = [item["docs"] for item in labeled_entries]

    fig_height = max(6, len(stages) * 0.35)
    fig, ax = plt.subplots(figsize=(width, fig_height))
    bars = ax.barh(range(len(stages) - 1, -1, -1), totals, color="#4C72B0")
    ax.set_xlabel("Total resident seconds")
    ax.set_yticks(range(len(stages) - 1, -1, -1))
    ax.set_yticklabels(stages)
    if log_scale:
        ax.set_xscale("log")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, avg, doc_cnt in zip(bars, avgs, docs):
        width_val = bar.get_width()
        ax.text(
            width_val,
            bar.get_y() + bar.get_height() / 2,
            f"{width_val:.1f}s (avg {avg:.2f}s, docs {doc_cnt})",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stage resident times from results.json")
    parser.add_argument("results", type=Path, help="Path to results.json artifact")
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
    return parser.parse_args()


def main():
    args = parse_args()
    build_plot(
        results_path=args.results,
        pipeline_yaml=args.pipeline,
        top_n=args.top_n,
        log_scale=args.log_scale,
        width=args.width,
        keep_nested=args.keep_nested,
        sort_mode=args.sort,
        summary_rows=args.summary_rows,
        exclude_network=args.exclude_network,
    )


if __name__ == "__main__":
    main()
