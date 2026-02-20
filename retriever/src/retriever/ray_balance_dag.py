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


@dataclass
class Variant:
    run_id: str
    # Matrix input metadata columns
    pdf_workers: int
    pdf_bs: int
    ocr_workers: int
    ocr_bs: int
    embed_workers: int
    embed_bs: int
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
        "pdf_bs",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "return_code",
        "pages",
        "wall_secs",
        "ingest_secs",
        "pages_per_sec_total",
        "pages_per_sec_ingest",
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
            if col in {"wall_secs", "ingest_secs", "pages_per_sec_total", "pages_per_sec_ingest"}:
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

    # Matrix from the tuning discussion.
    variants = [
        Variant("A", pdf_workers=12, pdf_bs=2, ocr_workers=1, ocr_bs=16, embed_workers=1, embed_bs=256),
        Variant("B", pdf_workers=12, pdf_bs=2, ocr_workers=1, ocr_bs=32, embed_workers=1, embed_bs=256),
        Variant("C", pdf_workers=12, pdf_bs=2, ocr_workers=2, ocr_bs=16, embed_workers=1, embed_bs=256),
        Variant("D", pdf_workers=10, pdf_bs=2, ocr_workers=2, ocr_bs=16, embed_workers=1, embed_bs=256),
        Variant("E", pdf_workers=10, pdf_bs=2, ocr_workers=2, ocr_bs=16, embed_workers=2, embed_bs=256),
    ]

    base_cmd = shlex.split(args.runner) + ["-m", args.module]
    rows: list[dict[str, Any]] = []

    for variant in variants:
        cmd = base_cmd + [str(input_dir), "--query-csv", str(args.query_csv), "--no-recall-details"]
        if variant.ray_address:
            cmd += ["--ray-address", variant.ray_address]
        if variant.start_ray:
            cmd += ["--start-ray"]

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

        row: dict[str, Any] = {
            "run_id": variant.run_id,
            "pdf_workers": variant.pdf_workers,
            "pdf_bs": variant.pdf_bs,
            "ocr_workers": variant.ocr_workers,
            "ocr_bs": variant.ocr_bs,
            "embed_workers": variant.embed_workers,
            "embed_bs": variant.embed_bs,
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
        rows.append(row)

        (logs_dir / f"{variant.run_id}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (logs_dir / f"{variant.run_id}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        print(
            f"Run {variant.run_id}: rc={proc.returncode}, pages={row['pages']}, "
            f"wall={format_float(row['wall_secs'])}s, "
            f"pps_total={format_float(row['pages_per_sec_total'])}"
        )

    fieldnames = [
        "run_id",
        "pdf_workers",
        "pdf_bs",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
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
