#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DONE_RE = re.compile(
    r"\[done\]\s+(?P<files>\d+)\s+files,\s+(?P<pages>\d+)\s+pages\s+in\s+(?P<secs>[0-9.]+)s"
)
RECALL_LINE_RE = re.compile(r"^\s{2}(?P<key>[^:]+):\s*(?P<val>-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)


@dataclass
class Variant:
    run_id: str
    # Matrix input metadata columns
    pdf_workers: int
    pdf_num_cpus: float
    pdf_bs: int
    ocr_workers: int
    ocr_bs: int
    embed_workers: int
    embed_bs: int
    gpu_page_elements: float
    gpu_ocr: float
    gpu_embed: float
    # Supported by batch_pipeline.py today
    ray_address: str | None = None
    start_ray: bool = False


def parse_done_metrics(stdout: str) -> dict[str, Any]:
    match = DONE_RE.search(stdout)
    if not match:
        return {"files": None, "pages": None, "ingest_secs": None}
    return {
        "files": int(match.group("files")),
        "pages": int(match.group("pages")),
        "ingest_secs": float(match.group("secs")),
    }


def _sanitize_metric_key(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return f"recall_{s}" if s else "recall_metric"


def parse_recall_metrics(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if "Recall metrics (matching retriever.recall.core):" not in stdout:
        return out
    for m in RECALL_LINE_RE.finditer(stdout):
        key = _sanitize_metric_key(m.group("key"))
        try:
            out[key] = float(m.group("val"))
        except ValueError:
            continue
    return out


def format_float(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def print_matrix(rows: list[dict[str, Any]]) -> None:
    cols = [
        "run_id",
        "pdf_workers",
        "pdf_num_cpus",
        "pdf_bs",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "gpu_page_elements",
        "gpu_ocr",
        "gpu_embed",
        "return_code",
        "pages",
        "wall_secs",
        "ingest_secs",
        "pages_per_sec_total",
        "pages_per_sec_ingest",
        "recall_recall_1",
        "recall_recall_5",
        "recall_recall_10",
    ]

    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in cols}
    header = " | ".join(col.ljust(widths[col]) for col in cols)
    sep = "-+-".join("-" * widths[col] for col in cols)

    print("\nResults Matrix")
    print(header)
    print(sep)
    for row in rows:
        vals: list[str] = []
        for col in cols:
            value = row.get(col, "")
            if col in {
                "pdf_num_cpus",
                "gpu_page_elements",
                "gpu_ocr",
                "gpu_embed",
                "wall_secs",
                "ingest_secs",
                "pages_per_sec_total",
                "pages_per_sec_ingest",
                "recall_recall_1",
                "recall_recall_5",
                "recall_recall_10",
            }:
                value = format_float(value, 3)
            vals.append(str(value).ljust(widths[col]))
        print(" | ".join(vals))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run retriever.examples.batch_pipeline across variants and report throughput."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing PDFs.")
    parser.add_argument("--output-csv", default="batch_matrix_results.csv", help="CSV output path.")
    parser.add_argument("--query-csv", default="bo767_query_gt.csv", help="Query CSV for batch_pipeline.")
    parser.add_argument(
        "--runner",
        default="uv run python",
        help="Command prefix used to invoke Python (default: 'uv run python').",
    )
    parser.add_argument(
        "--module",
        default="retriever.examples.batch_pipeline",
        help="Python module to run (default: retriever.examples.batch_pipeline).",
    )
    parser.add_argument(
        "--logs-dir",
        default="batch_matrix_logs",
        help="Directory for per-run stdout/stderr logs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    # Expanded matrix:
    # - vary per-stage GPU fractions in 0.25 increments
    # - vary CPU allocation and worker count for PDF extraction
    gpu_triplets = [
        (0.25, 0.50, 0.25),
        (0.25, 0.75, 0.50),
        (0.50, 0.75, 0.50),
        (0.50, 1.00, 0.50),
        (0.75, 1.00, 0.75),
        (0.75, 1.25, 0.75),
        (1.00, 1.00, 1.00),
        (1.00, 1.25, 1.00),
    ]
    cpu_setups = [
        (8, 1.0),
        (12, 2.0),
        (16, 3.0),
    ]

    variants: list[Variant] = []
    idx = 0
    for pdf_workers, pdf_num_cpus in cpu_setups:
        for gpu_page_elements, gpu_ocr, gpu_embed in gpu_triplets:
            idx += 1
            variants.append(
                Variant(
                    run_id=f"V{idx:02d}",
                    pdf_workers=pdf_workers,
                    pdf_num_cpus=pdf_num_cpus,
                    pdf_bs=4,
                    ocr_workers=1 if gpu_ocr < 1.0 else 2,
                    ocr_bs=16,
                    embed_workers=1 if gpu_embed <= 0.5 else 2,
                    embed_bs=256,
                    gpu_page_elements=gpu_page_elements,
                    gpu_ocr=gpu_ocr,
                    gpu_embed=gpu_embed,
                )
            )

    base_cmd = shlex.split(args.runner) + ["-m", args.module]
    rows: list[dict[str, Any]] = []

    for variant in variants:
        cmd = base_cmd + [str(input_dir), "--query-csv", str(args.query_csv), "--no-recall-details"]
        if variant.ray_address:
            cmd += ["--ray-address", variant.ray_address]
        if variant.start_ray:
            cmd += ["--start-ray"]
        cmd += [
            "--pdf-extract-workers",
            str(variant.pdf_workers),
            "--pdf-extract-num-cpus",
            str(variant.pdf_num_cpus),
            "--pdf-extract-batch-size",
            str(variant.pdf_bs),
            "--ocr-workers",
            str(variant.ocr_workers),
            "--ocr-batch-size",
            str(variant.ocr_bs),
            "--embed-workers",
            str(variant.embed_workers),
            "--embed-batch-size",
            str(variant.embed_bs),
            "--gpu-page-elements",
            str(variant.gpu_page_elements),
            "--gpu-ocr",
            str(variant.gpu_ocr),
            "--gpu-embed",
            str(variant.gpu_embed),
        ]

        print(f"\n=== Run {variant.run_id} ===")
        print("CMD:", " ".join(shlex.quote(x) for x in cmd))

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
        wall_secs = time.perf_counter() - t0

        metrics = parse_done_metrics(proc.stdout)
        pages = metrics["pages"]
        ingest_secs = metrics["ingest_secs"]
        recall_metrics = parse_recall_metrics(proc.stdout)

        row: dict[str, Any] = {
            "run_id": variant.run_id,
            "pdf_workers": variant.pdf_workers,
            "pdf_num_cpus": variant.pdf_num_cpus,
            "pdf_bs": variant.pdf_bs,
            "ocr_workers": variant.ocr_workers,
            "ocr_bs": variant.ocr_bs,
            "embed_workers": variant.embed_workers,
            "embed_bs": variant.embed_bs,
            "gpu_page_elements": variant.gpu_page_elements,
            "gpu_ocr": variant.gpu_ocr,
            "gpu_embed": variant.gpu_embed,
            "ray_address": variant.ray_address,
            "start_ray": variant.start_ray,
            "return_code": proc.returncode,
            "files": metrics["files"],
            "pages": pages,
            "wall_secs": wall_secs,
            "ingest_secs": ingest_secs,
            "pages_per_sec_total": (pages / wall_secs) if (pages and wall_secs > 0) else None,
            "pages_per_sec_ingest": (pages / ingest_secs) if (pages and ingest_secs and ingest_secs > 0) else None,
        }
        row.update(recall_metrics)
        rows.append(row)

        (logs_dir / f"{variant.run_id}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (logs_dir / f"{variant.run_id}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        print(
            f"Run {variant.run_id}: rc={proc.returncode}, pages={row['pages']}, "
            f"wall={format_float(row['wall_secs'])}s, "
            f"pps_total={format_float(row['pages_per_sec_total'])}, "
            f"recall@10={format_float(row.get('recall_recall_10'))}"
        )

    base_fieldnames = [
        "run_id",
        "pdf_workers",
        "pdf_num_cpus",
        "pdf_bs",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "gpu_page_elements",
        "gpu_ocr",
        "gpu_embed",
        "ray_address",
        "start_ray",
        "return_code",
        "files",
        "pages",
        "wall_secs",
        "ingest_secs",
        "pages_per_sec_total",
        "pages_per_sec_ingest",
    ]
    recall_fieldnames = sorted(
        {
            k
            for row in rows
            for k in row.keys()
            if k.startswith("recall_") and k not in {"recall_recall_1", "recall_recall_5", "recall_recall_10"}
        }
    )
    common_recall_fields = [k for k in ["recall_recall_1", "recall_recall_5", "recall_recall_10"] if any(k in r for r in rows)]
    fieldnames = base_fieldnames + common_recall_fields + recall_fieldnames
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print_matrix(rows)
    print(f"\nCSV written to: {output_csv}")
    print(f"Logs written to: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
