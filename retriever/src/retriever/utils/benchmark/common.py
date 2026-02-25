# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import ray
import ray.data as rd
import typer

from retriever.pdf.extract import PDFExtractionActor
from retriever.pdf.split import split_pdf_batch


@dataclass(frozen=True)
class ThroughputResult:
    workers: int
    batch_size: int
    rows: int
    elapsed_seconds: float
    rows_per_second: float


def parse_csv_ints(value: str, *, name: str) -> List[int]:
    out: List[int] = []
    for part in str(value or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError as e:
            raise typer.BadParameter(f"{name} must be a comma-separated list of ints, got {value!r}") from e
        if n <= 0:
            raise typer.BadParameter(f"{name} entries must be > 0, got {n}")
        out.append(n)
    if not out:
        raise typer.BadParameter(f"{name} cannot be empty")
    return sorted(set(out))


def maybe_init_ray(ray_address: Optional[str]) -> None:
    if not ray.is_initialized():
        ray.init(address=ray_address or "local", ignore_reinit_error=True)


def read_pdf_bytes(pdf_path: Path) -> bytes:
    p = pdf_path.expanduser().resolve()
    if not p.is_file():
        raise typer.BadParameter(f"PDF path does not exist: {p}")
    return p.read_bytes()


def make_seed_split_row(pdf_path: Path) -> Dict[str, Any]:
    pdf_bytes = read_pdf_bytes(pdf_path)
    return {"path": str(pdf_path.expanduser().resolve()), "bytes": pdf_bytes}


def make_seed_extraction_row(pdf_path: Path) -> Dict[str, Any]:
    split_df = split_pdf_batch(pd.DataFrame([make_seed_split_row(pdf_path)]))
    if split_df.empty:
        raise RuntimeError("PDF split returned no rows; cannot benchmark extraction stage.")
    row = split_df.iloc[0].to_dict()
    return row


def make_seed_page_elements_row(pdf_path: Path, *, dpi: int) -> Dict[str, Any]:
    extract_row = make_seed_extraction_row(pdf_path)
    extraction_actor = PDFExtractionActor(
        extract_text=True,
        # Force raster generation for seed rows. The extractor currently renders
        # images when one of these feature flags is enabled.
        extract_images=True,
        extract_tables=False,
        extract_charts=False,
        extract_infographics=False,
        extract_page_as_image=True,
        dpi=int(dpi),
    )
    extracted_df = extraction_actor(pd.DataFrame([extract_row]))
    if extracted_df is None or extracted_df.empty:
        raise RuntimeError("PDF extraction seed generation returned no rows.")
    out = extracted_df.iloc[0].to_dict()
    page_image = out.get("page_image") if isinstance(out, dict) else None
    if not (
        isinstance(page_image, dict) and isinstance(page_image.get("image_b64"), str) and page_image.get("image_b64")
    ):
        raise RuntimeError("Seed extraction row does not contain page_image.image_b64.")
    return out


def with_fake_detections(
    row: Dict[str, Any],
    *,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
) -> Dict[str, Any]:
    labels: List[str] = []
    if extract_tables:
        labels.append("table")
    if extract_charts:
        labels.append("chart")
    if extract_infographics:
        labels.append("infographic")
    if not labels:
        labels.append("table")

    detections: List[Dict[str, Any]] = []
    x1 = 0.05
    y1 = 0.05
    width = 0.6
    height = 0.25
    for idx, label in enumerate(labels):
        top = min(0.7, y1 + idx * 0.28)
        det = {
            "bbox_xyxy_norm": [x1, top, min(0.98, x1 + width), min(0.98, top + height)],
            "label": idx,
            "label_name": label,
            "score": 0.99,
        }
        detections.append(det)

    out = dict(row)
    out["page_elements_v3"] = {"detections": detections, "timing": {"seconds": 0.0}, "error": None}
    out["page_elements_v3_num_detections"] = len(detections)
    out["page_elements_v3_counts_by_label"] = {d["label_name"]: 1 for d in detections}
    return out


def benchmark_sweep(
    *,
    stage_name: str,
    seed_row: Dict[str, Any],
    rows: int,
    workers: Sequence[int],
    batch_sizes: Sequence[int],
    map_builder: Callable[[rd.Dataset, int, int], rd.Dataset],
) -> Tuple[ThroughputResult, List[ThroughputResult]]:
    all_results: List[ThroughputResult] = []
    best: Optional[ThroughputResult] = None

    base_items = [dict(seed_row) for _ in range(int(rows))]
    for worker_count in workers:
        for batch_size in batch_sizes:
            ds = rd.from_items(base_items)
            mapped = map_builder(ds, int(worker_count), int(batch_size))
            t0 = time.perf_counter()
            count = int(mapped.count())
            elapsed = max(time.perf_counter() - t0, 1e-9)
            rows_per_second = float(count) / elapsed
            result = ThroughputResult(
                workers=int(worker_count),
                batch_size=int(batch_size),
                rows=count,
                elapsed_seconds=float(elapsed),
                rows_per_second=float(rows_per_second),
            )
            all_results.append(result)
            typer.echo(
                f"[{stage_name}] workers={result.workers} batch_size={result.batch_size} "
                f"rows={result.rows} elapsed={result.elapsed_seconds:.3f}s rps={result.rows_per_second:.2f}"
            )
            if best is None or result.rows_per_second > best.rows_per_second:
                best = result

    if best is None:
        raise RuntimeError(f"No benchmark results were produced for stage {stage_name}")
    return best, all_results


def maybe_write_results_json(
    path: Optional[Path], *, best: ThroughputResult, results: Sequence[ThroughputResult]
) -> None:
    if path is None:
        return
    payload = {
        "best": {
            "workers": best.workers,
            "batch_size": best.batch_size,
            "rows": best.rows,
            "elapsed_seconds": best.elapsed_seconds,
            "rows_per_second": best.rows_per_second,
        },
        "results": [
            {
                "workers": r.workers,
                "batch_size": r.batch_size,
                "rows": r.rows,
                "elapsed_seconds": r.elapsed_seconds,
                "rows_per_second": r.rows_per_second,
            }
            for r in results
        ],
    }
    out = path.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"Wrote results JSON to {out}")
