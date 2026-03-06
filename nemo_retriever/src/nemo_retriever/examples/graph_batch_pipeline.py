# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Graph-based batch ingestion pipeline using the Graph + RayDataExecutor.

Replicates every stage from ``batch_pipeline.py`` but defines them
declaratively as ``Node`` objects in a ``Graph``.  The ``RayDataExecutor``
builds the full lazy Ray Data plan (one ``map_batches`` per node) and
materialises only the leaf — letting Ray Data fuse, pipeline, and
back-pressure across all stages in a single execution.

Stages (PDF path, all options enabled):

  DocToPdf -> PDFSplit -> PDFExtraction -> PageElements -> OCR
           -> explode_content_to_rows -> Embed -> LanceDB Write

Run with:
    uv run python -m nemo_retriever.examples.graph_batch_pipeline <input-dir>
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from collections import defaultdict
from datetime import timedelta
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import ray
import ray.data as rd
import typer

from nemo_retriever.graph.graph import Graph
from nemo_retriever.graph.node import Node
from nemo_retriever.graph.ray_executor import RayDataExecutor
from nemo_retriever.ocr.ocr import OCRActor
from nemo_retriever.page_elements import PageElementDetectionActor
from nemo_retriever.params import (
    EmbedParams,
    PdfSplitParams,
    VdbUploadParams,
)
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score
from nemo_retriever.utils.convert import DocToPdfConversionActor

logger = logging.getLogger(__name__)


def _lancedb():
    """Import lancedb lazily to avoid fork warnings during early process setup."""
    return import_module("lancedb")

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_files(input_dir: Path, pattern: str) -> list[str]:
    """Expand *pattern* under *input_dir* and return absolute paths."""
    matches = glob.glob(str(input_dir / pattern), recursive=True)
    files = sorted(os.path.abspath(p) for p in matches if os.path.isfile(p))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_dir / pattern}")
    return files


def _ensure_lancedb_table(uri: str, table_name: str) -> None:
    """Create LanceDB table with expected schema if it does not exist."""
    from importlib import import_module

    lancedb = import_module("lancedb")
    import pyarrow as pa

    Path(uri).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2048)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
    )
    empty = pa.table({f.name: [] for f in schema}, schema=schema)
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    """Estimate pages processed by counting unique (source_id, page_number) pairs."""
    try:
        db = _lancedb().connect(uri)
        table = db.open_table(table_name)
    except Exception:
        return None

    try:
        df = table.to_pandas()[["source_id", "page_number"]]
        return int(df.dropna(subset=["source_id", "page_number"]).drop_duplicates().shape[0])
    except Exception:
        try:
            return int(table.count_rows())
        except Exception:
            return None


def _collect_detection_summary(uri: str, table_name: str) -> Optional[dict]:
    """Collect per-model detection totals deduped by (source_id, page_number)."""
    try:
        db = _lancedb().connect(uri)
        table = db.open_table(table_name)
        df = table.to_pandas()[["source_id", "page_number", "metadata"]]
    except Exception:
        return None

    per_page: dict[tuple[str, int], dict] = {}
    for row in df.itertuples(index=False):
        source_id = str(getattr(row, "source_id", "") or "")
        page_number = _to_int(getattr(row, "page_number", -1), default=-1)
        key = (source_id, page_number)

        raw_metadata = getattr(row, "metadata", None)
        meta: dict = {}
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:
                meta = {}

        entry = per_page.setdefault(
            key,
            {
                "page_elements_total": 0,
                "ocr_table_total": 0,
                "ocr_chart_total": 0,
                "ocr_infographic_total": 0,
                "page_elements_by_label": defaultdict(int),
            },
        )

        pe_total = _to_int(meta.get("page_elements_v3_num_detections"), default=0)
        entry["page_elements_total"] = max(entry["page_elements_total"], pe_total)

        ocr_table = _to_int(meta.get("ocr_table_detections"), default=0)
        ocr_chart = _to_int(meta.get("ocr_chart_detections"), default=0)
        ocr_infographic = _to_int(meta.get("ocr_infographic_detections"), default=0)
        entry["ocr_table_total"] = max(entry["ocr_table_total"], ocr_table)
        entry["ocr_chart_total"] = max(entry["ocr_chart_total"], ocr_chart)
        entry["ocr_infographic_total"] = max(entry["ocr_infographic_total"], ocr_infographic)

        label_counts = meta.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                if not isinstance(label, str):
                    continue
                entry["page_elements_by_label"][label] = max(
                    entry["page_elements_by_label"][label],
                    _to_int(count, default=0),
                )

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    page_elements_total = 0
    ocr_table_total = 0
    ocr_chart_total = 0
    ocr_infographic_total = 0
    for page_entry in per_page.values():
        page_elements_total += int(page_entry["page_elements_total"])
        ocr_table_total += int(page_entry["ocr_table_total"])
        ocr_chart_total += int(page_entry["ocr_chart_total"])
        ocr_infographic_total += int(page_entry["ocr_infographic_total"])
        for label, count in page_entry["page_elements_by_label"].items():
            pe_by_label_totals[label] += int(count)

    return {
        "pages_seen": int(len(per_page)),
        "page_elements_v3_total_detections": int(page_elements_total),
        "page_elements_v3_counts_by_label": dict(sorted(pe_by_label_totals.items())),
        "ocr_table_total_detections": int(ocr_table_total),
        "ocr_chart_total_detections": int(ocr_chart_total),
        "ocr_infographic_total_detections": int(ocr_infographic_total),
    }


def _print_detection_summary(summary: Optional[dict]) -> None:
    if summary is None:
        print("Detection summary: unavailable (could not read LanceDB metadata).")
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


def _print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print(f"Pages/sec: unavailable (could not estimate processed pages). Ingest time: {ingest_elapsed_s:.2f}s")
        return
    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only; excludes Ray startup and recall): {pps:.2f}")


def _gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def _is_hit_at_k(golden_key: str, retrieved_keys: list[str], k: int) -> bool:
    doc, page = _gold_to_doc_page(golden_key)
    specific_page = f"{doc}_{page}"
    entire_document = f"{doc}_-1"
    top = (retrieved_keys or [])[: int(k)]
    return (specific_page in top) or (entire_document in top)


def _hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:
    try:
        res = json.loads(hit.get("metadata", "{}"))
        source = json.loads(hit.get("source", "{}"))
    except Exception:
        return None, None

    source_id = source.get("source_id")
    page_number = res.get("page_number")
    if not source_id or page_number is None:
        return None, float(hit.get("_distance")) if "_distance" in hit else None

    key = f"{Path(str(source_id)).stem}_{page_number}"
    dist = float(hit["_distance"]) if "_distance" in hit else float(hit["_score"]) if "_score" in hit else None
    return key, dist


def _create_lancedb_index(
    lancedb_uri: str,
    table_name: str,
    *,
    index_type: str = "IVF_HNSW_SQ",
    metric: str = "l2",
    num_partitions: int = 16,
    num_sub_vectors: int = 256,
    hybrid: bool = False,
    text_column: str = "text",
    fts_language: str = "English",
) -> None:
    """Create the LanceDB vector index after streaming writes finish."""
    try:
        import lancedb  # type: ignore
    except Exception as e:
        print(f"Warning: lancedb not available for index creation: {e}")
        return

    try:
        db = lancedb.connect(uri=lancedb_uri)
        table = db.open_table(table_name)
        n_vecs = table.count_rows()
    except Exception as e:
        print(f"Warning: could not open LanceDB table for indexing: {e}")
        return

    if n_vecs < 2:
        print("Skipping LanceDB index creation (not enough vectors).")
        return

    k = int(num_partitions)
    if k >= n_vecs:
        k = max(1, n_vecs - 1)

    try:
        table.create_index(
            index_type=index_type,
            metric=metric,
            num_partitions=k,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
        )
    except TypeError:
        table.create_index(vector_column_name="vector")
    except Exception as e:
        print(f"Warning: failed to create LanceDB index (continuing without index): {e}")

    if hybrid:
        try:
            table.create_fts_index(text_column, language=fts_language)
        except Exception as e:
            print(
                f"Warning: FTS index creation failed on column {text_column!r} (continuing with vector-only): {e}"
            )

    for index_stub in table.list_indices():
        table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

    print(f"Wrote {n_vecs} rows to LanceDB uri={lancedb_uri!r} table={table_name!r}")


# ---------------------------------------------------------------------------
# Graph construction — mirrors BatchIngestor.extract / embed / vdb_upload
# ---------------------------------------------------------------------------


def build_pdf_graph(
    *,
    # PDF stages
    pdf_split_batch_size: int = 1,
    pdf_extract_batch_size: int = 4,
    pdf_extract_workers: int = 12,
    pdf_extract_num_cpus: float = 2.0,
    extract_kwargs: dict[str, Any],
    # Page-elements detection
    detect_kwargs: dict[str, Any],
    page_elements_batch_size: int = 24,
    page_elements_workers: int = 1,
    page_elements_cpus_per_actor: float = 1.0,
    gpu_page_elements: float = 0.5,
    # OCR
    ocr_kwargs: dict[str, Any],
    ocr_batch_size: int = 16,
    ocr_workers: int = 1,
    ocr_cpus_per_actor: float = 1.0,
    gpu_ocr: float = 1.0,
    # Embedding
    embed_params: EmbedParams,
    embed_batch_size: int = 256,
    embed_workers: int = 1,
    embed_cpus_per_actor: float = 1.0,
    gpu_embed: float = 0.5,
    embed_modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
    # VDB upload
    vdb_params: VdbUploadParams,
) -> Graph:
    """Build a sequential PDF ingestion graph matching ``BatchIngestor``.

    Stages:
      1. DocToPdf          – DOCX/PPTX → PDF conversion (passthrough for PDFs)
      2. PDFSplit           – multi-page PDF → one row per page
      3. PDFExtraction      – render page image + extract text (TaskPool, CPU)
      4. PageElements       – YOLOX page-element detection (ActorPool, GPU)
      5. OCR                – Nemotron OCR for tables/charts/infographics (ActorPool, GPU)
      6. explode_content    – explode structured elements to individual rows (CPU)
      7. Embed              – text/image embedding (ActorPool, GPU)
      8. LanceDB Write      – streaming write to LanceDB (ActorPool, CPU)
    """
    from nemo_retriever.ingest_modes.batch import _BatchEmbedActor, _LanceDBWriteActor
    from nemo_retriever.ingest_modes.inprocess import explode_content_to_rows

    g = Graph()

    # ---- 1. DocToPdf conversion (CPU-only, batch_size=1) ----
    g.add(Node(
        "doc_to_pdf",
        DocToPdfConversionActor,
        map_kwargs={
            "batch_size": 1,
            "batch_format": "pandas",
            "num_cpus": 1,
            "num_gpus": 0,
        },
    ))

    # ---- 2. PDF split ----
    g.add(Node(
        "pdf_split",
        PDFSplitActor(split_params=PdfSplitParams(
            start_page=extract_kwargs.get("start_page"),
            end_page=extract_kwargs.get("end_page"),
        )),
        map_kwargs={
            "batch_size": pdf_split_batch_size,
            "batch_format": "pandas",
            "num_cpus": 1,
            "num_gpus": 0,
        },
    ))

    # ---- 3. PDF extraction (CPU TaskPool) ----
    g.add(Node(
        "pdf_extraction",
        PDFExtractionActor(**extract_kwargs),
        map_kwargs={
            "batch_size": pdf_extract_batch_size,
            "batch_format": "pandas",
            "num_cpus": pdf_extract_num_cpus,
            "num_gpus": 0,
            "compute": rd.TaskPoolStrategy(size=pdf_extract_workers),
        },
    ))

    # ---- 4. Page-element detection (GPU ActorPool) ----
    g.add(Node(
        "page_elements",
        PageElementDetectionActor,
        constructor_kwargs=dict(detect_kwargs),
        map_kwargs={
            "batch_size": page_elements_batch_size,
            "batch_format": "pandas",
            "num_cpus": page_elements_cpus_per_actor,
            "num_gpus": gpu_page_elements,
            "compute": rd.ActorPoolStrategy(size=page_elements_workers),
        },
    ))

    # ---- 5. OCR (GPU ActorPool) — conditional on extract_tables/charts/infographics ----
    has_ocr = any(
        ocr_kwargs.get(k)
        for k in ("extract_tables", "extract_charts", "extract_infographics")
    )
    if has_ocr:
        g.add(Node(
            "ocr",
            OCRActor,
            constructor_kwargs=dict(ocr_kwargs),
            map_kwargs={
                "batch_size": ocr_batch_size,
                "batch_format": "pandas",
                "num_cpus": ocr_cpus_per_actor,
                "num_gpus": gpu_ocr,
                "compute": rd.ActorPoolStrategy(size=ocr_workers),
            },
        ))

    # ---- 6. Explode content to individual embedding rows (CPU) ----
    _text_mod = text_elements_modality or embed_modality
    _struct_mod = structured_elements_modality or embed_modality
    explode_fn = partial(
        explode_content_to_rows,
        modality=embed_modality,
        text_elements_modality=_text_mod,
        structured_elements_modality=_struct_mod,
    )
    g.add(Node(
        "explode_content",
        explode_fn,
        map_kwargs={
            "batch_size": embed_batch_size,
            "batch_format": "pandas",
            "num_cpus": 1,
            "num_gpus": 0,
        },
    ))

    # ---- 7. Embedding (GPU ActorPool) ----
    endpoint = (
        embed_params.embed_invoke_url or ""
    ).strip()
    embed_gpu = 0 if endpoint else gpu_embed
    g.add(Node(
        "embed",
        _BatchEmbedActor,
        constructor_kwargs={"params": embed_params},
        map_kwargs={
            "batch_size": embed_batch_size,
            "batch_format": "pandas",
            "num_cpus": embed_cpus_per_actor,
            "num_gpus": embed_gpu,
            "compute": rd.ActorPoolStrategy(size=embed_workers),
        },
    ))

    # ---- 8. LanceDB streaming write (CPU ActorPool) ----
    g.add(Node(
        "vdb_write",
        _LanceDBWriteActor,
        constructor_kwargs={"params": vdb_params},
        map_kwargs={
            "batch_format": "pandas",
            "num_cpus": 1,
            "num_gpus": 0,
            "compute": rd.ActorPoolStrategy(size=1),
        },
    ))

    return g


# ---------------------------------------------------------------------------
# CLI — mirrors batch_pipeline.py options
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDF files to ingest.",
        path_type=Path,
        exists=True,
    ),
    glob_pattern: str = typer.Option(
        "*.pdf",
        "--glob",
        help="Glob pattern for files inside input_dir.",
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="Ray cluster address (e.g. 'auto'). Omit for local Ray.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
    ),
    # Recall evaluation
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help=(
            "Path to query CSV for recall evaluation. Default: bo767_query_gt.csv "
            "(current directory). Recall is skipped if the file does not exist."
        ),
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help=(
            "Do not print per-query retrieval details (query, gold, hits). "
            "Only the missed-gold summary and recall metrics are printed."
        ),
    ),
    # PDF stages
    pdf_split_batch_size: int = typer.Option(1, "--pdf-split-batch-size", min=1),
    pdf_extract_batch_size: int = typer.Option(4, "--pdf-extract-batch-size", min=1),
    pdf_extract_workers: int = typer.Option(12, "--pdf-extract-workers", min=1),
    pdf_extract_num_cpus: float = typer.Option(2.0, "--pdf-extract-num-cpus", min=0.1),
    # Page-elements
    page_elements_batch_size: int = typer.Option(24, "--page-elements-batch-size", min=1),
    page_elements_workers: int = typer.Option(1, "--page-elements-workers", min=1),
    page_elements_cpus_per_actor: float = typer.Option(1.0, "--page-elements-cpus-per-actor", min=0.1),
    gpu_page_elements: float = typer.Option(0.5, "--gpu-page-elements", min=0.0),
    page_elements_invoke_url: Optional[str] = typer.Option(None, "--page-elements-invoke-url"),
    # OCR
    ocr_batch_size: int = typer.Option(16, "--ocr-batch-size", min=1),
    ocr_workers: int = typer.Option(1, "--ocr-workers", min=1),
    ocr_cpus_per_actor: float = typer.Option(1.0, "--ocr-cpus-per-actor", min=0.1),
    gpu_ocr: float = typer.Option(1.0, "--gpu-ocr", min=0.0),
    ocr_invoke_url: Optional[str] = typer.Option(None, "--ocr-invoke-url"),
    extract_tables: bool = typer.Option(True, "--extract-tables/--no-extract-tables"),
    extract_charts: bool = typer.Option(True, "--extract-charts/--no-extract-charts"),
    extract_infographics: bool = typer.Option(True, "--extract-infographics/--no-extract-infographics"),
    # Embedding
    embed_batch_size: int = typer.Option(256, "--embed-batch-size", min=1),
    embed_workers: int = typer.Option(1, "--embed-workers", min=1),
    embed_cpus_per_actor: float = typer.Option(1.0, "--embed-cpus-per-actor", min=0.1),
    gpu_embed: float = typer.Option(0.5, "--gpu-embed", min=0.0),
    embed_invoke_url: Optional[str] = typer.Option(None, "--embed-invoke-url"),
    embed_model_name: str = typer.Option("nvidia/llama-3.2-nv-embedqa-1b-v2", "--embed-model-name"),
    embed_modality: str = typer.Option("text", "--embed-modality"),
    text_elements_modality: Optional[str] = typer.Option(None, "--text-elements-modality"),
    structured_elements_modality: Optional[str] = typer.Option(None, "--structured-elements-modality"),
    # LanceDB
    lancedb_uri: str = typer.Option(LANCEDB_URI, "--lancedb-uri"),
    lancedb_table: str = typer.Option(LANCEDB_TABLE, "--lancedb-table"),
    hybrid: bool = typer.Option(False, "--hybrid/--no-hybrid"),
) -> None:
    """Ingest PDFs via a Graph-based pipeline on the RayDataExecutor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # -- Resolve input files --
    files = _resolve_files(Path(input_dir), glob_pattern)
    logger.info("Found %d file(s) matching %s/%s", len(files), input_dir, glob_pattern)

    # -- Init Ray --
    os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
    ray.init(address=ray_address or "local", ignore_reinit_error=True, log_to_driver=ray_log_to_driver)

    ctx = rd.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    # -- Build initial Ray dataset --
    ds = rd.read_binary_files(files, include_paths=True)

    # -- Prepare LanceDB --
    lancedb_uri_abs = str(Path(lancedb_uri).expanduser().resolve())
    _ensure_lancedb_table(lancedb_uri_abs, lancedb_table)

    # -- Suppress GPU for remote-endpoint stages (mirrors batch_pipeline.py) --
    if page_elements_invoke_url and gpu_page_elements != 0.0:
        logger.warning("--page-elements-invoke-url set; forcing --gpu-page-elements to 0.0")
        gpu_page_elements = 0.0
    if ocr_invoke_url and gpu_ocr != 0.0:
        logger.warning("--ocr-invoke-url set; forcing --gpu-ocr to 0.0")
        gpu_ocr = 0.0
    if embed_invoke_url and gpu_embed != 0.0:
        logger.warning("--embed-invoke-url set; forcing --gpu-embed to 0.0")
        gpu_embed = 0.0

    # -- Build kwargs for each stage (mirrors BatchIngestor.extract) --
    extract_kwargs: dict[str, Any] = {
        "extract_text": True,
        "extract_page_as_image": True,
        "dpi": 200,
    }

    detect_kwargs: dict[str, Any] = {}
    if page_elements_invoke_url:
        detect_kwargs["invoke_url"] = page_elements_invoke_url

    ocr_flags: dict[str, Any] = {}
    if extract_tables:
        ocr_flags["extract_tables"] = True
    if extract_charts:
        ocr_flags["extract_charts"] = True
    if extract_infographics:
        ocr_flags["extract_infographics"] = True
    if ocr_invoke_url:
        ocr_flags["invoke_url"] = ocr_invoke_url

    embed_params = EmbedParams(
        model_name=embed_model_name,
        embed_invoke_url=embed_invoke_url,
        embed_modality=embed_modality,
        text_elements_modality=text_elements_modality,
        structured_elements_modality=structured_elements_modality,
    )

    vdb_params = VdbUploadParams(
        lancedb={
            "lancedb_uri": lancedb_uri_abs,
            "table_name": lancedb_table,
            "overwrite": True,
            "create_index": True,
            "hybrid": hybrid,
        }
    )

    # -- Build the graph --
    graph = build_pdf_graph(
        pdf_split_batch_size=pdf_split_batch_size,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_workers=pdf_extract_workers,
        pdf_extract_num_cpus=pdf_extract_num_cpus,
        extract_kwargs=extract_kwargs,
        detect_kwargs=detect_kwargs,
        page_elements_batch_size=page_elements_batch_size,
        page_elements_workers=page_elements_workers,
        page_elements_cpus_per_actor=page_elements_cpus_per_actor,
        gpu_page_elements=gpu_page_elements,
        ocr_kwargs=ocr_flags,
        ocr_batch_size=ocr_batch_size,
        ocr_workers=ocr_workers,
        ocr_cpus_per_actor=ocr_cpus_per_actor,
        gpu_ocr=gpu_ocr,
        embed_params=embed_params,
        embed_batch_size=embed_batch_size,
        embed_workers=embed_workers,
        embed_cpus_per_actor=embed_cpus_per_actor,
        gpu_embed=gpu_embed,
        embed_modality=embed_modality,
        text_elements_modality=text_elements_modality,
        structured_elements_modality=structured_elements_modality,
        vdb_params=vdb_params,
    )

    logger.info(
        "Graph built with %d node(s): %s",
        len(graph.nodes),
        " -> ".join(graph.nodes.keys()),
    )

    # -- Execute via RayDataExecutor --
    # Per-node map_kwargs already carry batch_size, num_cpus, num_gpus, compute.
    # The executor defaults provide only batch_format as a fallback.
    executor = RayDataExecutor(init_ray=False, map_kwargs={"batch_format": "pandas"})

    logger.info("Starting graph execution...")
    t0 = time.perf_counter()
    outputs = executor.run(graph, ds)
    elapsed = time.perf_counter() - t0

    # -- Report results --
    leaf_node = list(outputs.keys())[-1]
    leaf_ds = outputs[leaf_node]
    row_count = leaf_ds.count()

    print(f"\nGraph execution complete in {elapsed:.2f}s")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Final node ({leaf_node}): {row_count} rows")

    ingest_elapsed_s = elapsed
    processed_pages = _estimate_processed_pages(lancedb_uri_abs, lancedb_table)
    detection_summary = _collect_detection_summary(lancedb_uri_abs, lancedb_table)
    print("Extraction complete.")
    _print_detection_summary(detection_summary)

    # Create LanceDB vector index after all streaming writes are complete.
    _create_lancedb_index(
        lancedb_uri_abs,
        lancedb_table,
        hybrid=hybrid,
    )

    ray.shutdown()

    # ---------------------------------------------------------------------------
    # Recall calculation (optional)
    # ---------------------------------------------------------------------------
    query_csv = Path(query_csv)
    if not query_csv.exists():
        print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
        return

    db = _lancedb().connect(lancedb_uri_abs)
    table = None
    open_err: Optional[Exception] = None
    for _ in range(3):
        try:
            table = db.open_table(lancedb_table)
            open_err = None
            break
        except Exception as e:
            open_err = e
            _ensure_lancedb_table(lancedb_uri_abs, lancedb_table)
            time.sleep(2)
    if table is None:
        raise RuntimeError(
            f"Recall stage requires LanceDB table {lancedb_table!r} at {lancedb_uri_abs!r}, "
            f"but it was not found."
        ) from open_err
    try:
        if int(table.count_rows()) == 0:
            print(f"LanceDB table {lancedb_table!r} exists but is empty; skipping recall evaluation.")
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return
    except Exception:
        pass
    unique_basenames = table.to_pandas()["pdf_basename"].unique()
    print(f"Unique basenames: {unique_basenames}")

    # Resolve the HF model ID for recall query embedding so aliases
    # (e.g. "nemo_retriever_v1") map to the correct model.
    from nemo_retriever.model import resolve_embed_model

    _recall_model = resolve_embed_model(str(embed_model_name))

    cfg = RecallConfig(
        lancedb_uri=str(lancedb_uri_abs),
        lancedb_table=str(lancedb_table),
        embedding_model=_recall_model,
        embedding_http_endpoint=embed_invoke_url,
        top_k=10,
        ks=(1, 5, 10),
        hybrid=hybrid,
    )

    _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

    if not no_recall_details:
        print("\nPer-query retrieval details:")
    missed_gold: list[tuple[str, str]] = []
    ext = ".pdf"
    for i, (q, g, hits) in enumerate(
        zip(
            _df_query["query"].astype(str).tolist(),
            _gold,
            _raw_hits,
        )
    ):
        doc, page = _gold_to_doc_page(g)

        scored_hits: list[tuple[str, float | None]] = []
        for h in hits:
            key, dist = _hit_key_and_distance(h)
            if key:
                scored_hits.append((key, dist))

        top_keys = [k for (k, _d) in scored_hits]
        hit = _is_hit_at_k(g, top_keys, cfg.top_k)

        if not no_recall_details:
            print(f"\nQuery {i}: {q}")
            print(f"  Gold: {g}  (file: {doc}{ext}, page: {page})")
            print(f"  Hit@{cfg.top_k}: {hit}")
            print("  Top hits:")
            if not scored_hits:
                print("    (no hits)")
            else:
                for rank, (key, dist) in enumerate(scored_hits[: int(cfg.top_k)], start=1):
                    if dist is None:
                        print(f"    {rank:02d}. {key}")
                    else:
                        print(f"    {rank:02d}. {key}  distance={dist:.6f}")

        if not hit:
            missed_gold.append((f"{doc}{ext}", str(page)))

    missed_unique = sorted(set(missed_gold), key=lambda x: (x[0], x[1]))
    print("\nMissed gold (unique doc/page):")
    if not missed_unique:
        print("  (none)")
    else:
        for doc_page, page in missed_unique:
            print(f"  {doc_page} page {page}")
    print(f"\nTotal missed: {len(missed_unique)} / {len(_gold)}")

    print("\nRecall metrics (matching nemo_retriever.recall.core):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    _print_pages_per_second(processed_pages, ingest_elapsed_s)


if __name__ == "__main__":
    app()
