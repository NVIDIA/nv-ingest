#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import shlex
import shutil
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
    pdf_split_bs: int
    pdf_bs: int
    page_elements_bs: int
    page_elements_workers: int
    ocr_workers: int
    ocr_bs: int
    embed_workers: int
    embed_bs: int
    page_elements_cpus_per_actor: float
    ocr_cpus_per_actor: float
    embed_cpus_per_actor: float
    gpu_page_elements: float
    gpu_ocr: float
    gpu_embed: float
    # Supported by batch_pipeline.py today
    ray_address: str | None = None
    start_ray: bool = False


MATRIX_FIELDS = [
    "run_id",
    "pdf_workers",
    "pdf_num_cpus",
    "pdf_split_bs",
    "pdf_bs",
    "page_elements_bs",
    "page_elements_workers",
    "ocr_workers",
    "ocr_bs",
    "embed_workers",
    "embed_bs",
    "page_elements_cpus_per_actor",
    "ocr_cpus_per_actor",
    "embed_cpus_per_actor",
    "gpu_page_elements",
    "gpu_ocr",
    "gpu_embed",
    "ray_address",
    "start_ray",
]


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


def _parse_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _is_valid_variant(v: Variant) -> bool:
    # Reject invalid fractional GPU requests above 1.0 (e.g. 1.25), which many
    # Ray clusters do not accept per actor.
    for gpu in (v.gpu_page_elements, v.gpu_ocr, v.gpu_embed):
        if gpu < 0:
            return False
        if gpu > 1.0 and abs(gpu - round(gpu)) > 1e-9:
            return False
    return True


def build_default_variants() -> list[Variant]:
    # Compact DOE-style matrix (< 1000 rows):
    # 1) baseline
    # 2) one-factor sweeps around baseline
    # 3) targeted interaction grids for likely bottlenecks
    baseline = {
        "pdf_workers": 8,
        "pdf_num_cpus": 2.0,
        "pdf_split_bs": 1,
        "pdf_bs": 16,
        "page_elements_bs": 16,
        "page_elements_workers": 1,
        "ocr_workers": 2,
        "ocr_bs": 16,
        "embed_workers": 1,
        "embed_bs": 256,
        "page_elements_cpus_per_actor": 1.0,
        "ocr_cpus_per_actor": 2.0,
        "embed_cpus_per_actor": 1.0,
        "gpu_page_elements": 0.5,
        "gpu_ocr": 1.0,
        "gpu_embed": 0.5,
    }

    variants: list[Variant] = []
    seen: set[tuple[Any, ...]] = set()

    def _key(v: dict[str, Any]) -> tuple[Any, ...]:
        return (
            v["pdf_workers"],
            v["pdf_num_cpus"],
            v["pdf_split_bs"],
            v["pdf_bs"],
            v["page_elements_bs"],
            v["page_elements_workers"],
            v["ocr_workers"],
            v["ocr_bs"],
            v["embed_workers"],
            v["embed_bs"],
            v["page_elements_cpus_per_actor"],
            v["ocr_cpus_per_actor"],
            v["embed_cpus_per_actor"],
            v["gpu_page_elements"],
            v["gpu_ocr"],
            v["gpu_embed"],
        )

    def add_variant(**overrides: Any) -> None:
        cfg = dict(baseline)
        cfg.update(overrides)
        k = _key(cfg)
        if k in seen:
            return
        seen.add(k)
        v = Variant(
                run_id=f"V{len(variants) + 1:05d}",
                pdf_workers=int(cfg["pdf_workers"]),
                pdf_num_cpus=float(cfg["pdf_num_cpus"]),
                pdf_split_bs=int(cfg["pdf_split_bs"]),
                pdf_bs=int(cfg["pdf_bs"]),
                page_elements_bs=int(cfg["page_elements_bs"]),
                page_elements_workers=int(cfg["page_elements_workers"]),
                ocr_workers=int(cfg["ocr_workers"]),
                ocr_bs=int(cfg["ocr_bs"]),
                embed_workers=int(cfg["embed_workers"]),
                embed_bs=int(cfg["embed_bs"]),
                page_elements_cpus_per_actor=float(cfg["page_elements_cpus_per_actor"]),
                ocr_cpus_per_actor=float(cfg["ocr_cpus_per_actor"]),
                embed_cpus_per_actor=float(cfg["embed_cpus_per_actor"]),
                gpu_page_elements=float(cfg["gpu_page_elements"]),
                gpu_ocr=float(cfg["gpu_ocr"]),
                gpu_embed=float(cfg["gpu_embed"]),
        )
        if _is_valid_variant(v):
            variants.append(v)

    # 1) Baseline
    add_variant()

    # 2) One-factor sweeps
    for x in [4, 8, 12, 16]:
        add_variant(pdf_workers=x)
    for x in [1.0, 2.0, 3.0, 4.0]:
        add_variant(pdf_num_cpus=x)
    for x in [1, 4, 8]:
        add_variant(pdf_split_bs=x)
    for x in [8, 16, 24, 32]:
        add_variant(pdf_bs=x)
    for x in [8, 16, 24, 32]:
        add_variant(page_elements_bs=x)
    for x in [1, 2, 3]:
        add_variant(page_elements_workers=x)
    for x in [1, 2, 3]:
        add_variant(ocr_workers=x)
    for x in [8, 16, 24, 32]:
        add_variant(ocr_bs=x)
    for x in [1, 2, 3]:
        add_variant(embed_workers=x)
    for x in [128, 256, 512, 768]:
        add_variant(embed_bs=x)
    for x in [1.0, 2.0, 4.0]:
        add_variant(page_elements_cpus_per_actor=x)
    for x in [1.0, 2.0, 4.0]:
        add_variant(ocr_cpus_per_actor=x)
    for x in [1.0, 2.0, 4.0]:
        add_variant(embed_cpus_per_actor=x)
    for x in [0.25, 0.5, 0.75]:
        add_variant(gpu_page_elements=x)
    for x in [0.75, 1.0]:
        add_variant(gpu_ocr=x)
    for x in [0.25, 0.5, 0.75]:
        add_variant(gpu_embed=x)

    # 3) Targeted interactions
    for ocr_bs, ocr_workers, gpu_ocr in itertools.product([8, 16, 24, 32], [1, 2, 3], [0.75, 1.0]):
        add_variant(ocr_bs=ocr_bs, ocr_workers=ocr_workers, gpu_ocr=gpu_ocr)

    for embed_bs, embed_workers, gpu_embed in itertools.product([128, 256, 512, 768], [1, 2, 3], [0.25, 0.5, 0.75]):
        add_variant(embed_bs=embed_bs, embed_workers=embed_workers, gpu_embed=gpu_embed)

    for page_elements_bs, page_elements_workers, gpu_page_elements in itertools.product(
        [8, 16, 24, 32], [1, 2, 3], [0.25, 0.5, 0.75]
    ):
        add_variant(
            page_elements_bs=page_elements_bs,
            page_elements_workers=page_elements_workers,
            gpu_page_elements=gpu_page_elements,
        )

    for pdf_workers, pdf_num_cpus, pdf_bs in itertools.product([4, 8, 12, 16], [1.0, 2.0, 3.0, 4.0], [8, 16, 24, 32]):
        add_variant(pdf_workers=pdf_workers, pdf_num_cpus=pdf_num_cpus, pdf_bs=pdf_bs)

    for pe_cpu, ocr_cpu, embed_cpu in itertools.product([1.0, 2.0, 4.0], [1.0, 2.0, 4.0], [1.0, 2.0, 4.0]):
        add_variant(
            page_elements_cpus_per_actor=pe_cpu,
            ocr_cpus_per_actor=ocr_cpu,
            embed_cpus_per_actor=embed_cpu,
        )

    for pdf_bs, ocr_bs, embed_bs in itertools.product([8, 16, 24, 32], [8, 16, 24, 32], [128, 256, 512, 768]):
        add_variant(pdf_bs=pdf_bs, ocr_bs=ocr_bs, embed_bs=embed_bs)

    return variants


def write_matrix_csv(variants: list[Variant], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_variants = [v for v in variants if _is_valid_variant(v)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        for v in valid_variants:
            writer.writerow(
                {
                    "run_id": v.run_id,
                    "pdf_workers": v.pdf_workers,
                    "pdf_num_cpus": v.pdf_num_cpus,
                    "pdf_split_bs": v.pdf_split_bs,
                    "pdf_bs": v.pdf_bs,
                    "page_elements_bs": v.page_elements_bs,
                    "page_elements_workers": v.page_elements_workers,
                    "ocr_workers": v.ocr_workers,
                    "ocr_bs": v.ocr_bs,
                    "embed_workers": v.embed_workers,
                    "embed_bs": v.embed_bs,
                    "page_elements_cpus_per_actor": v.page_elements_cpus_per_actor,
                    "ocr_cpus_per_actor": v.ocr_cpus_per_actor,
                    "embed_cpus_per_actor": v.embed_cpus_per_actor,
                    "gpu_page_elements": v.gpu_page_elements,
                    "gpu_ocr": v.gpu_ocr,
                    "gpu_embed": v.gpu_embed,
                    "ray_address": v.ray_address or "",
                    "start_ray": str(v.start_ray).lower(),
                }
            )


def load_variants_from_csv(path: Path) -> list[Variant]:
    required = {
        "pdf_workers",
        "pdf_num_cpus",
        "pdf_split_bs",
        "pdf_bs",
        "page_elements_bs",
        "page_elements_workers",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "page_elements_cpus_per_actor",
        "ocr_cpus_per_actor",
        "embed_cpus_per_actor",
        "gpu_page_elements",
        "gpu_ocr",
        "gpu_embed",
    }
    variants: list[Variant] = []
    invalid_rows = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Matrix CSV has no header: {path}")
        missing = sorted(required.difference(set(reader.fieldnames)))
        if missing:
            raise ValueError(f"Matrix CSV is missing required columns: {missing}")

        for idx, row in enumerate(reader, start=1):
            run_id = (row.get("run_id") or "").strip() or f"R{idx:05d}"
            ray_address_raw = (row.get("ray_address") or "").strip()
            raw = Variant(
                run_id=run_id,
                pdf_workers=int(row["pdf_workers"]),
                pdf_num_cpus=float(row["pdf_num_cpus"]),
                pdf_split_bs=int(row.get("pdf_split_bs") or 1),
                pdf_bs=int(row["pdf_bs"]),
                page_elements_bs=int(row.get("page_elements_bs") or 24),
                page_elements_workers=int(row.get("page_elements_workers") or 1),
                ocr_workers=int(row["ocr_workers"]),
                ocr_bs=int(row["ocr_bs"]),
                embed_workers=int(row["embed_workers"]),
                embed_bs=int(row["embed_bs"]),
                page_elements_cpus_per_actor=float(row.get("page_elements_cpus_per_actor") or 1.0),
                ocr_cpus_per_actor=float(row.get("ocr_cpus_per_actor") or 1.0),
                embed_cpus_per_actor=float(row.get("embed_cpus_per_actor") or 1.0),
                gpu_page_elements=float(row["gpu_page_elements"]),
                gpu_ocr=float(row["gpu_ocr"]),
                gpu_embed=float(row["gpu_embed"]),
                ray_address=ray_address_raw if ray_address_raw else None,
                start_ray=_parse_bool(row.get("start_ray"), default=False),
            )
            if _is_valid_variant(raw):
                variants.append(raw)
            else:
                invalid_rows += 1
    if invalid_rows:
        print(f"Skipped invalid variant rows from CSV: {invalid_rows}")
    return variants


def format_float(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


def print_matrix(rows: list[dict[str, Any]]) -> None:
    cols = [
        "matrix_row",
        "run_id",
        "pdf_workers",
        "pdf_num_cpus",
        "pdf_split_bs",
        "pdf_bs",
        "page_elements_bs",
        "page_elements_workers",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "page_elements_cpus_per_actor",
        "ocr_cpus_per_actor",
        "embed_cpus_per_actor",
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
                "page_elements_cpus_per_actor",
                "ocr_cpus_per_actor",
                "embed_cpus_per_actor",
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
    parser.add_argument(
        "--lancedb-uri",
        default="lancedb",
        help="LanceDB URI/path reused by each run (cleaned before each run).",
    )
    parser.add_argument(
        "--matrix-csv",
        default=None,
        help=(
            "Optional external matrix CSV path. If omitted, a default matrix is generated "
            "using built-in cpu/gpu settings and the requested batch-size ranges."
        ),
    )
    parser.add_argument(
        "--write-default-matrix-csv",
        default=None,
        help="Optional path to write the generated default matrix CSV.",
    )
    parser.add_argument(
        "--exit-after-writing-matrix",
        action="store_true",
        help="Write matrix CSV then exit without running any jobs.",
    )
    parser.add_argument(
        "--row-start",
        type=int,
        default=1,
        help="1-based inclusive start row from matrix to run.",
    )
    parser.add_argument(
        "--row-end",
        type=int,
        default=None,
        help="1-based inclusive end row from matrix to run. Defaults to final row.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    lancedb_uri_path = Path(args.lancedb_uri).expanduser().resolve()

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    if args.matrix_csv:
        matrix_path = Path(args.matrix_csv).resolve()
        if not matrix_path.exists():
            print(f"ERROR: matrix CSV does not exist: {matrix_path}", file=sys.stderr)
            return 2
        variants = load_variants_from_csv(matrix_path)
        print(f"Loaded matrix CSV from: {matrix_path}")
        print(f"Loaded matrix variant rows: {len(variants)}")
    else:
        variants = build_default_variants()
        print("Loaded matrix source: generated default variants")
        print(f"Loaded matrix variant rows: {len(variants)}")

    if args.write_default_matrix_csv:
        write_path = Path(args.write_default_matrix_csv).resolve()
        # Write whichever matrix is currently active so sharding is reproducible.
        write_matrix_csv(variants, write_path)
        print(f"Wrote matrix CSV ({len(variants)} rows): {write_path}")
        if args.exit_after_writing_matrix:
            return 0

    total_rows = len(variants)
    if total_rows == 0:
        print("ERROR: matrix contains zero rows.", file=sys.stderr)
        return 2

    row_start = int(args.row_start)
    row_end = int(args.row_end) if args.row_end is not None else total_rows
    if row_start < 1:
        print("--row-start must be >= 1", file=sys.stderr)
        return 2
    if row_end < row_start:
        print("--row-end must be >= --row-start", file=sys.stderr)
        return 2
    if row_end > total_rows:
        print(f"--row-end ({row_end}) exceeds matrix size ({total_rows})", file=sys.stderr)
        return 2

    # Preserve original matrix row numbers for distributed sharding visibility.
    selected: list[tuple[int, Variant]] = [
        (idx, variants[idx - 1]) for idx in range(row_start, row_end + 1)
    ]

    print(f"Loaded matrix rows: {total_rows}")
    print(f"Running row range: [{row_start}, {row_end}] ({len(selected)} rows)")

    base_cmd = shlex.split(args.runner) + ["-m", args.module]
    rows: list[dict[str, Any]] = []

    for matrix_row, variant in selected:
        # Ensure runs are isolated: remove prior LanceDB artifacts before each run.
        if lancedb_uri_path.exists():
            if lancedb_uri_path.is_dir():
                shutil.rmtree(lancedb_uri_path)
            else:
                lancedb_uri_path.unlink()

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
            "--pdf-split-batch-size",
            str(variant.pdf_split_bs),
            "--pdf-extract-batch-size",
            str(variant.pdf_bs),
            "--page-elements-batch-size",
            str(variant.page_elements_bs),
            "--page-elements-workers",
            str(variant.page_elements_workers),
            "--ocr-workers",
            str(variant.ocr_workers),
            "--ocr-batch-size",
            str(variant.ocr_bs),
            "--embed-workers",
            str(variant.embed_workers),
            "--embed-batch-size",
            str(variant.embed_bs),
            "--page-elements-cpus-per-actor",
            str(variant.page_elements_cpus_per_actor),
            "--ocr-cpus-per-actor",
            str(variant.ocr_cpus_per_actor),
            "--embed-cpus-per-actor",
            str(variant.embed_cpus_per_actor),
            "--gpu-page-elements",
            str(variant.gpu_page_elements),
            "--gpu-ocr",
            str(variant.gpu_ocr),
            "--gpu-embed",
            str(variant.gpu_embed),
            "--lancedb-uri",
            str(lancedb_uri_path),
            "--runtime-metrics-dir",
            str(logs_dir / "runtime_metrics"),
            "--runtime-metrics-prefix",
            f"{matrix_row:05d}_{variant.run_id}",
        ]

        print(f"\n=== Run {variant.run_id} (row {matrix_row}) ===")
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
        recall_ran = bool(recall_metrics)
        effective_return_code = proc.returncode if (proc.returncode != 0 or recall_ran) else 98
        failure_reason = ""
        if proc.returncode != 0:
            failure_reason = f"subprocess_exit_{proc.returncode}"
        elif not recall_ran:
            failure_reason = "missing_recall_metrics"

        row: dict[str, Any] = {
            "matrix_row": matrix_row,
            "run_id": variant.run_id,
            "pdf_workers": variant.pdf_workers,
            "pdf_num_cpus": variant.pdf_num_cpus,
            "pdf_split_bs": variant.pdf_split_bs,
            "pdf_bs": variant.pdf_bs,
            "page_elements_bs": variant.page_elements_bs,
            "page_elements_workers": variant.page_elements_workers,
            "ocr_workers": variant.ocr_workers,
            "ocr_bs": variant.ocr_bs,
            "embed_workers": variant.embed_workers,
            "embed_bs": variant.embed_bs,
            "page_elements_cpus_per_actor": variant.page_elements_cpus_per_actor,
            "ocr_cpus_per_actor": variant.ocr_cpus_per_actor,
            "embed_cpus_per_actor": variant.embed_cpus_per_actor,
            "gpu_page_elements": variant.gpu_page_elements,
            "gpu_ocr": variant.gpu_ocr,
            "gpu_embed": variant.gpu_embed,
            "ray_address": variant.ray_address,
            "start_ray": variant.start_ray,
            "return_code": effective_return_code,
            "recall_ran": recall_ran,
            "failure_reason": failure_reason,
            "files": metrics["files"],
            "pages": pages,
            "wall_secs": wall_secs,
            "ingest_secs": ingest_secs,
            "pages_per_sec_total": (pages / wall_secs) if (pages and wall_secs > 0) else None,
            "pages_per_sec_ingest": (pages / ingest_secs) if (pages and ingest_secs and ingest_secs > 0) else None,
        }
        row.update(recall_metrics)
        rows.append(row)

        (logs_dir / f"{matrix_row:05d}_{variant.run_id}.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
        (logs_dir / f"{matrix_row:05d}_{variant.run_id}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        print(
            f"Run {variant.run_id}: rc={effective_return_code}, pages={row['pages']}, "
            f"wall={format_float(row['wall_secs'])}s, "
            f"pps_total={format_float(row['pages_per_sec_total'])}, "
            f"recall@10={format_float(row.get('recall_recall_10'))}"
        )

    base_fieldnames = [
        "matrix_row",
        "run_id",
        "pdf_workers",
        "pdf_num_cpus",
        "pdf_split_bs",
        "pdf_bs",
        "page_elements_bs",
        "page_elements_workers",
        "ocr_workers",
        "ocr_bs",
        "embed_workers",
        "embed_bs",
        "page_elements_cpus_per_actor",
        "ocr_cpus_per_actor",
        "embed_cpus_per_actor",
        "gpu_page_elements",
        "gpu_ocr",
        "gpu_embed",
        "ray_address",
        "start_ray",
        "return_code",
        "recall_ran",
        "failure_reason",
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
