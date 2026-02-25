"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m retriever.examples.batch_pipeline <input-dir>
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, TextIO

import lancedb
import ray
import typer
from retriever import create_ingestor
from retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


class _TeeStream:
    """Write stream output to terminal and optional log file."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(getattr(self._primary, "fileno")())

    def writable(self) -> bool:
        return bool(getattr(self._primary, "writable", lambda: True)())

    @property
    def encoding(self) -> str:
        return str(getattr(self._primary, "encoding", "utf-8"))


def _configure_logging(log_file: Optional[Path]) -> tuple[Optional[TextIO], TextIO, TextIO]:
    """Configure root logging; optionally tee stdout/stderr into one file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if log_file is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return None, original_stdout, original_stderr

    target = Path(log_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fh = open(target, "a", encoding="utf-8", buffering=1)

    # Tee stdout/stderr so print(), tracebacks, and Ray driver-forwarded logs
    # all land in the same place while still showing on the console.
    sys.stdout = _TeeStream(sys.__stdout__, fh)
    sys.stderr = _TeeStream(sys.__stderr__, fh)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("Writing combined pipeline logs to %s", str(target))
    return fh, original_stdout, original_stderr


def _estimate_processed_pages(uri: str, table_name: str) -> Optional[int]:
    """
    Estimate pages processed by counting unique (source_id, page_number) pairs.

    Falls back to table row count if page-level fields are unavailable.
    """
    try:
        db = lancedb.connect(uri)
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


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _collect_detection_summary(uri: str, table_name: str) -> Optional[dict]:
    """
    Collect per-model detection totals deduped by (source_id, page_number).

    Counts are read from LanceDB row `metadata`, which is populated during batch
    ingestion by the Ray write stage.
    """
    try:
        db = lancedb.connect(uri)
        table = db.open_table(table_name)
        df = table.to_pandas()[["source_id", "page_number", "metadata"]]
    except Exception:
        return None

    # Deduplicate exploded rows by page key; keep max per-page counts.
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


def _write_detection_summary(path: Path, summary: Optional[dict]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = summary if summary is not None else {"error": "Detection summary unavailable."}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print("Pages/sec: unavailable (could not estimate processed pages). " f"Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only; excludes Ray startup and recall): {pps:.2f}")


def _ensure_lancedb_table(uri: str, table_name: str) -> None:
    """
    Ensure the local LanceDB URI exists and table can be opened.

    Creates an empty table with the expected schema if it does not exist yet.
    """
    # Local path URI in this pipeline.
    Path(uri).mkdir(parents=True, exist_ok=True)

    db = lancedb.connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    import pyarrow as pa  # type: ignore

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
    empty = pa.table(
        {
            "vector": [],
            "pdf_page": [],
            "filename": [],
            "pdf_basename": [],
            "page_number": [],
            "source_id": [],
            "path": [],
            "text": [],
            "metadata": [],
            "source": [],
        },
        schema=schema,
    )
    db.create_table(table_name, data=empty, schema=schema, mode="create")


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


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
        exists=True,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', or 'doc'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice).",  # noqa: E501
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    start_ray: bool = typer.Option(
        False,
        "--start-ray",
        help=(
            "Start a Ray head node (ray start --head) and connect to it. "
            "Dashboard at http://127.0.0.1:8265. Ignores --ray-address."
        ),
    ),
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
    pdf_extract_workers: int = typer.Option(
        12,
        "--pdf-extract-workers",
        min=1,
        help="Number of CPU workers for PDF extraction stage.",
    ),
    pdf_extract_num_cpus: float = typer.Option(
        2.0,
        "--pdf-extract-num-cpus",
        min=0.1,
        help="CPUs reserved per PDF extraction task.",
    ),
    pdf_extract_batch_size: int = typer.Option(
        4,
        "--pdf-extract-batch-size",
        min=1,
        help="Batch size for PDF extraction stage.",
    ),
    pdf_split_batch_size: int = typer.Option(
        1,
        "--pdf-split-batch-size",
        min=1,
        help="Batch size for PDF split stage.",
    ),
    page_elements_batch_size: int = typer.Option(
        24,
        "--page-elements-batch-size",
        min=1,
        help="Ray Data batch size for page-elements stage.",
    ),
    ocr_workers: int = typer.Option(
        1,
        "--ocr-workers",
        min=1,
        help="Actor count for OCR stage.",
    ),
    page_elements_workers: int = typer.Option(
        1,
        "--page-elements-workers",
        min=1,
        help="Actor count for page-elements stage.",
    ),
    ocr_batch_size: int = typer.Option(
        16,
        "--ocr-batch-size",
        min=1,
        help="Ray Data batch size for OCR stage.",
    ),
    page_elements_cpus_per_actor: float = typer.Option(
        1.0,
        "--page-elements-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per page-elements actor.",
    ),
    ocr_cpus_per_actor: float = typer.Option(
        1.0,
        "--ocr-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per OCR actor.",
    ),
    embed_workers: int = typer.Option(
        1,
        "--embed-workers",
        min=1,
        help="Actor count for embedding stage.",
    ),
    embed_batch_size: int = typer.Option(
        256,
        "--embed-batch-size",
        min=1,
        help="Ray Data batch size for embedding stage.",
    ),
    embed_cpus_per_actor: float = typer.Option(
        1.0,
        "--embed-cpus-per-actor",
        min=0.1,
        help="CPUs reserved per embedding actor.",
    ),
    gpu_page_elements: float = typer.Option(
        0.5,
        "--gpu-page-elements",
        min=0.0,
        help="GPUs reserved per page-elements actor.",
    ),
    gpu_ocr: float = typer.Option(
        1.0,
        "--gpu-ocr",
        min=0.0,
        help="GPUs reserved per OCR actor.",
    ),
    gpu_embed: float = typer.Option(
        0.5,
        "--gpu-embed",
        min=0.0,
        help="GPUs reserved per embedding actor.",
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Optional remote endpoint URL for page-elements model inference.",
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None,
        "--ocr-invoke-url",
        help="Optional remote endpoint URL for OCR model inference.",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Optional remote endpoint URL for embedding model inference.",
    ),
    runtime_metrics_dir: Optional[Path] = typer.Option(
        None,
        "--runtime-metrics-dir",
        path_type=Path,
        file_okay=False,
        dir_okay=True,
        help="Optional directory where Ray runtime metrics are written per run.",
    ),
    runtime_metrics_prefix: Optional[str] = typer.Option(
        None,
        "--runtime-metrics-prefix",
        help="Optional filename prefix for per-run metrics artifacts.",
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        path_type=Path,
        dir_okay=False,
        help="Optional file to collect all pipeline + Ray driver logs for this run.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver (recommended with --log-file).",
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Optional JSON file path to write end-of-run detection counts summary.",
    ),
) -> None:
    log_handle, original_stdout, original_stderr = _configure_logging(log_file)
    try:
        os.environ.setdefault("NEMOTRON_OCR_MODEL_DIR", str(Path.cwd() / "nemotron-ocr-v1"))
        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        # Use an absolute path so driver and Ray actors resolve the same LanceDB URI.
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        # Remote endpoints don't need local model GPUs for their stage.
        if page_elements_invoke_url and float(gpu_page_elements) != 0.0:
            print(
                "[WARN] --page-elements-invoke-url is set; forcing --gpu-page-elements from "
                f"{float(gpu_page_elements):.3f} to 0.0"
            )
            gpu_page_elements = 0.0

        if ocr_invoke_url and float(gpu_ocr) != 0.0:
            print("[WARN] --ocr-invoke-url is set; forcing --gpu-ocr from " f"{float(gpu_ocr):.3f} to 0.0")
            gpu_ocr = 0.0

        if embed_invoke_url and float(gpu_embed) != 0.0:
            print("[WARN] --embed-invoke-url is set; forcing --gpu-embed from " f"{float(gpu_embed):.3f} to 0.0")
            gpu_embed = 0.0

        # Resolve Ray: start a head node, connect to given address, or run in-process
        if start_ray:
            subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
            ray_address = "auto"

        input_dir = Path(input_dir)
        if input_type == "txt":
            glob_pattern = str(input_dir / "*.txt")
            ingestor = create_ingestor(
                run_mode="batch",
                ray_address=ray_address,
                ray_log_to_driver=ray_log_to_driver,
            )
            ingestor = (
                ingestor.files(glob_pattern)
                .extract_txt(max_tokens=512, overlap_tokens=0)
                .embed(model_name="nemo_retriever_v1", embed_invoke_url=embed_invoke_url)
                .vdb_upload(
                    lancedb_uri=lancedb_uri,
                    table_name=LANCEDB_TABLE,
                    overwrite=True,
                    create_index=True,
                    hybrid=bool(hybrid),
                )
            )
        elif input_type == "html":
            glob_pattern = str(input_dir / "*.html")
            ingestor = create_ingestor(
                run_mode="batch",
                ray_address=ray_address,
                ray_log_to_driver=ray_log_to_driver,
            )
            ingestor = (
                ingestor.files(glob_pattern)
                .extract_html(max_tokens=512, overlap_tokens=0)
                .embed(model_name="nemo_retriever_v1", embed_invoke_url=embed_invoke_url)
                .vdb_upload(
                    lancedb_uri=lancedb_uri,
                    table_name=LANCEDB_TABLE,
                    overwrite=True,
                    create_index=True,
                    hybrid=bool(hybrid),
                )
            )
        elif input_type == "doc":
            # DOCX/PPTX: same pipeline as PDF; DocToPdfConversionActor converts before split.
            doc_globs = [str(input_dir / "*.docx"), str(input_dir / "*.pptx")]
            ingestor = create_ingestor(
                run_mode="batch",
                ray_address=ray_address,
                ray_log_to_driver=ray_log_to_driver,
            )
            ingestor = (
                ingestor.files(doc_globs)
                .extract(
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    debug_run_id=str(runtime_metrics_prefix or "unknown"),
                    pdf_extract_workers=int(pdf_extract_workers),
                    pdf_extract_num_cpus=float(pdf_extract_num_cpus),
                    pdf_split_batch_size=int(pdf_split_batch_size),
                    pdf_extract_batch_size=int(pdf_extract_batch_size),
                    page_elements_batch_size=int(page_elements_batch_size),
                    page_elements_workers=int(page_elements_workers),
                    detect_workers=int(ocr_workers),
                    detect_batch_size=int(ocr_batch_size),
                    page_elements_cpus_per_actor=float(page_elements_cpus_per_actor),
                    ocr_cpus_per_actor=float(ocr_cpus_per_actor),
                    gpu_page_elements=float(gpu_page_elements),
                    gpu_ocr=float(gpu_ocr),
                    gpu_embed=float(gpu_embed),
                    page_elements_invoke_url=page_elements_invoke_url,
                    ocr_invoke_url=ocr_invoke_url,
                )
                .embed(
                    model_name="nemo_retriever_v1",
                    embed_workers=int(embed_workers),
                    embed_batch_size=int(embed_batch_size),
                    embed_cpus_per_actor=float(embed_cpus_per_actor),
                    embed_invoke_url=embed_invoke_url,
                )
                .vdb_upload(
                    lancedb_uri=lancedb_uri,
                    table_name=LANCEDB_TABLE,
                    overwrite=True,
                    create_index=True,
                    hybrid=bool(hybrid),
                )
            )
        else:
            pdf_glob = str(input_dir / "*.pdf")
            ingestor = create_ingestor(
                run_mode="batch",
                ray_address=ray_address,
                ray_log_to_driver=ray_log_to_driver,
            )
            ingestor = (
                ingestor.files(pdf_glob)
                .extract(
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    debug_run_id=str(runtime_metrics_prefix or "unknown"),
                    pdf_extract_workers=int(pdf_extract_workers),
                    pdf_extract_num_cpus=float(pdf_extract_num_cpus),
                    pdf_split_batch_size=int(pdf_split_batch_size),
                    pdf_extract_batch_size=int(pdf_extract_batch_size),
                    page_elements_batch_size=int(page_elements_batch_size),
                    page_elements_workers=int(page_elements_workers),
                    detect_workers=int(ocr_workers),
                    detect_batch_size=int(ocr_batch_size),
                    page_elements_cpus_per_actor=float(page_elements_cpus_per_actor),
                    ocr_cpus_per_actor=float(ocr_cpus_per_actor),
                    gpu_page_elements=float(gpu_page_elements),
                    gpu_ocr=float(gpu_ocr),
                    gpu_embed=float(gpu_embed),
                    page_elements_invoke_url=page_elements_invoke_url,
                    ocr_invoke_url=ocr_invoke_url,
                )
                .embed(
                    model_name="nemo_retriever_v1",
                    embed_workers=int(embed_workers),
                    embed_batch_size=int(embed_batch_size),
                    embed_cpus_per_actor=float(embed_cpus_per_actor),
                    embed_invoke_url=embed_invoke_url,
                )
                .vdb_upload(
                    lancedb_uri=lancedb_uri,
                    table_name=LANCEDB_TABLE,
                    overwrite=True,
                    create_index=True,
                    hybrid=bool(hybrid),
                )
            )

        print("Running extraction...")
        ingest_start = time.perf_counter()
        ingestor.ingest(
            runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
            runtime_metrics_prefix=runtime_metrics_prefix,
        )
        ingest_elapsed_s = time.perf_counter() - ingest_start
        processed_pages = _estimate_processed_pages(lancedb_uri, LANCEDB_TABLE)
        detection_summary = _collect_detection_summary(lancedb_uri, LANCEDB_TABLE)
        print("Extraction complete.")
        _print_detection_summary(detection_summary)
        if detection_summary_file is not None:
            _write_detection_summary(detection_summary_file, detection_summary)
            print(f"Wrote detection summary JSON to {Path(detection_summary_file).expanduser().resolve()}")

        ray.shutdown()

        # ---------------------------------------------------------------------------
        # Recall calculation (optional)
        # ---------------------------------------------------------------------------
        query_csv = Path(query_csv)
        if not query_csv.exists():
            print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
            _print_pages_per_second(processed_pages, ingest_elapsed_s)
            return

        db = lancedb.connect(lancedb_uri)
        table = None
        open_err: Optional[Exception] = None
        for _ in range(3):
            try:
                table = db.open_table(LANCEDB_TABLE)
                open_err = None
                break
            except Exception as e:
                open_err = e
                # Create table if missing, then retry open.
                _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)
                time.sleep(2)
        if table is None:
            raise RuntimeError(
                f"Recall stage requires LanceDB table {LANCEDB_TABLE!r} at {lancedb_uri!r}, " f"but it was not found."
            ) from open_err
        try:
            if int(table.count_rows()) == 0:
                print(f"LanceDB table {LANCEDB_TABLE!r} exists but is empty; skipping recall evaluation.")
                _print_pages_per_second(processed_pages, ingest_elapsed_s)
                return
        except Exception:
            pass
        unique_basenames = table.to_pandas()["pdf_basename"].unique()
        print(f"Unique basenames: {unique_basenames}")

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            top_k=10,
            ks=(1, 5, 10),
            hybrid=bool(hybrid),
        )

        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

        if not no_recall_details:
            print("\nPer-query retrieval details:")
        missed_gold: list[tuple[str, str]] = []
        ext = (
            ".html"
            if input_type == "html"
            else (".txt" if input_type == "txt" else (".docx" if input_type == "doc" else ".pdf"))
        )
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

        print("\nRecall metrics (matching retriever.recall.core):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
    finally:
        # Restore real stdio before closing the mirror file so exception hooks
        # and late flushes never write to a closed stream wrapper.
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            try:
                log_handle.flush()
            finally:
                log_handle.close()


if __name__ == "__main__":
    app()
