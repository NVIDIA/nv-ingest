# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.batch_pipeline <input-dir>
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Optional, TextIO

import ray
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

logger = logging.getLogger(__name__)

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


def _lancedb():
    """Import lancedb lazily to avoid fork warnings during early process setup."""
    return import_module("lancedb")


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


def _configure_logging(log_file: Optional[Path], *, debug: bool = False) -> tuple[Optional[TextIO], TextIO, TextIO]:
    """Configure root logging; optionally tee stdout/stderr into one file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_level = logging.DEBUG if debug else logging.INFO
    if log_file is None:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
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
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("Writing combined pipeline logs to %s", str(target))
    return fh, original_stdout, original_stderr


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
        db = _lancedb().connect(uri)
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


def _extract_error_payloads(v: object) -> list[object]:
    """Recursively extract only error payloads from nested values."""
    payloads: list[object] = []
    if v is None:
        return payloads
    if isinstance(v, dict):
        if any(k in v for k in ("error", "errors", "exception", "traceback", "failed")):
            payloads.append(v)
        for item in v.values():
            payloads.extend(_extract_error_payloads(item))
        return payloads
    if isinstance(v, list):
        for item in v:
            payloads.extend(_extract_error_payloads(item))
        return payloads
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return payloads
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                payloads.extend(_extract_error_payloads(json.loads(s)))
                return payloads
            except Exception:
                pass
        low = s.lower()
        if any(tok in low for tok in ("error", "exception", "traceback", "failed")):
            payloads.append(s)
    return payloads


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

    db = _lancedb().connect(uri)
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
    ctx: typer.Context,
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable debug-level logging for this full pipeline run.",
    ),
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    detection_summary_file: Optional[Path] = typer.Option(
        None,
        "--detection-summary-file",
        path_type=Path,
        dir_okay=False,
        help="Optional JSON file path to write end-of-run detection counts summary.",
    ),
    recall_match_mode: str = typer.Option(
        "pdf_page",
        "--recall-match-mode",
        help="Recall match mode: 'pdf_page' or 'pdf_only'.",
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help=(
            "Do not print per-query retrieval details (query, gold, hits). "
            "Only the missed-gold summary and recall metrics are printed."
        ),
    ),
    embed_actors: Optional[int] = typer.Option(
        0,
        "--embed-actors",
        help="Actor count for embedding stage. Omit to use resource heuristic.",
    ),
    embed_batch_size: Optional[int] = typer.Option(
        0,
        "--embed-batch-size",
        help="Ray Data batch size for embedding stage.",
    ),
    embed_cpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--embed-cpus-per-actor",
        help="CPUs reserved per embedding actor. Omit to use resource heuristic.",
    ),
    embed_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--embed-gpus-per-actor",
        max=1.0,
        help="GPUs reserved per embedding actor. Omit to use resource heuristic.",
    ),
    embed_granularity: str = typer.Option(
        "element",
        "--embed-granularity",
        help="Embedding granularity: 'element' (one row per table/chart/text) or 'page' (one row per page).",
    ),
    graphic_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--graphic-elements-invoke-url",
        help="Optional remote endpoint URL for graphic-elements model inference.",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Optional remote endpoint URL for embedding model inference.",
    ),
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
    ),
    embed_modality: str = typer.Option(
        "text",
        "--embed-modality",
        help="Default embedding modality for all element types: "
        "'text', 'image', or 'text_image' ('image_text' is also accepted).",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', or 'doc'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice).",  # noqa: E501
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        path_type=Path,
        dir_okay=False,
        help="Optional file to collect all pipeline + Ray driver logs for this run.",
    ),
    # fmt: off
    nemotron_parse_actors: Optional[int] = typer.Option(
        0.0,
        "--nemotron-parse-actors",
        min=0.0,
        help=(
            "Actor count for Nemotron Parse stage "
            "(enables parse-only mode when > 0.0 with parse GPU/batch-size)."
        ),  # noqa: E501
    ),
    # fmt: on
    nemotron_parse_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--nemotron-parse-gpus-per-actor",
        min=0.0,
        max=1.0,
        help="GPUs reserved per Nemotron Parse actor.",
    ),
    nemotron_parse_batch_size: Optional[int] = typer.Option(
        0.0,
        "--nemotron-parse-batch-size",
        min=0.0,
        help="Ray Data batch size for Nemotron Parse stage "
        "(enables parse-only mode when > 0.0 with parse workers/GPU).",
    ),
    ocr_actors: Optional[int] = typer.Option(
        0,
        "--ocr-actors",
        help="Actor count for OCR stage. Omit to use resource heuristic.",
    ),
    ocr_batch_size: Optional[int] = typer.Option(
        0,
        "--ocr-batch-size",
        help="Batch size for OCR Ray stage and OCR inference batch size.",
    ),
    ocr_cpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--ocr-cpus-per-actor",
        help="CPUs reserved per OCR actor.",
    ),
    ocr_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--ocr-gpus-per-actor",
        min=0.0,
        max=1.0,
        help="GPUs reserved per OCR actor.",
    ),
    ocr_invoke_url: Optional[str] = typer.Option(
        None,
        "--ocr-invoke-url",
        help="Optional remote endpoint URL for OCR model inference.",
    ),
    page_elements_actors: Optional[int] = typer.Option(
        0,
        "--page-elements-actors",
        help="Actor count for page-elements stage. Omit to use resource heuristic.",
    ),
    page_elements_batch_size: Optional[int] = typer.Option(
        0,
        "--page-elements-batch-size",
        help="Page Elements batch size for both Ray stage and model inference batch size.",
    ),
    page_elements_cpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--page-elements-cpus-per-actor",
        help="CPUs reserved per page-elements actor.",
    ),
    page_elements_gpus_per_actor: Optional[float] = typer.Option(
        0.0,
        "--page-elements-gpus-per-actor",
        min=0.0,
        max=1.0,
        help="GPUs reserved per page-elements actor.",
    ),
    page_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--page-elements-invoke-url",
        help="Optional remote endpoint URL for page-elements model inference.",
    ),
    pdf_extract_batch_size: Optional[int] = typer.Option(
        0,
        "--pdf-extract-batch-size",
        help="Batch size for PDF extraction stage.",
    ),
    pdf_extract_cpus_per_task: Optional[float] = typer.Option(
        0.0,
        "--pdf-extract-cpus-per-task",
        help="CPUs reserved per PDF extraction task.",
    ),
    pdf_extract_tasks: Optional[int] = typer.Option(
        0,
        "--pdf-extract-tasks",
        help="Number of CPU tasks for PDF extraction stage.",
    ),
    pdf_split_batch_size: int = typer.Option(
        1,
        "--pdf-split-batch-size",
        min=1,
        help="Batch size for PDF split stage.",
    ),
    query_csv: Path = typer.Option(
        "./data/bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help=(
            "Path to query CSV for recall evaluation. Default: bo767_query_gt.csv "
            "(current directory). Recall is skipped if the file does not exist."
        ),
    ),
    ray_address: Optional[str] = typer.Option(
        None,
        "--ray-address",
        help="URL or address of a running Ray cluster (e.g. 'auto' or 'ray://host:10001'). Omit for in-process Ray.",
    ),
    ray_log_to_driver: bool = typer.Option(
        True,
        "--ray-log-to-driver/--no-ray-log-to-driver",
        help="Forward Ray worker logs to the driver (recommended with --log-file).",
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
    structured_elements_modality: Optional[str] = typer.Option(
        None,
        "--structured-elements-modality",
        help="Embedding modality override for table/chart/infographic rows. Falls back to --embed-modality.",
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None,
        "--text-elements-modality",
        help="Embedding modality override for page-text rows. Falls back to --embed-modality.",
    ),
    use_graphic_elements: bool = typer.Option(
        False,
        "--use-graphic-elements",
        help="Enable the combined graphic-elements + OCR stage for charts (requires extract_charts).",
    ),
    use_table_structure: bool = typer.Option(
        False,
        "--use-table-structure",
        help="Enable the combined table-structure + OCR stage for tables (requires extract_tables).",
    ),
    table_output_format: Optional[str] = typer.Option(
        None,
        "--table-output-format",
        help=(
            "Table output format: 'pseudo_markdown' (OCR-only) or 'markdown' "
            "(table-structure + OCR). Defaults to 'markdown' when table-structure "
            "is enabled, 'pseudo_markdown' otherwise."
        ),
    ),
    table_structure_invoke_url: Optional[str] = typer.Option(
        None,
        "--table-structure-invoke-url",
        help=(
            "Optional remote endpoint URL for table-structure model inference "
            "(used when --table-output-format=markdown)."
        ),
    ),
) -> None:
    log_handle, original_stdout, original_stderr = _configure_logging(log_file, debug=bool(debug))
    try:
        if recall_match_mode not in {"pdf_page", "pdf_only"}:
            raise ValueError(f"Unsupported --recall-match-mode: {recall_match_mode}")

        os.environ["RAY_LOG_TO_DRIVER"] = "1" if ray_log_to_driver else "0"
        # Use an absolute path so driver and Ray actors resolve the same LanceDB URI.
        lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())
        _ensure_lancedb_table(lancedb_uri, LANCEDB_TABLE)

        # Remote endpoints don't need local model GPUs for their stage.
        if page_elements_invoke_url and float(page_elements_gpus_per_actor) != 0.0:
            print(
                "[WARN] --page-elements-invoke-url is set; forcing --page-elements-gpus-per-actor from "
                f"{float(page_elements_gpus_per_actor):.3f} to 0.0"
            )
            page_elements_gpus_per_actor = 0.0

        if ocr_invoke_url and float(ocr_gpus_per_actor) != 0.0:
            print(
                "[WARN] --ocr-invoke-url is set; forcing --ocr-gpus-per-actor from "
                f"{float(ocr_gpus_per_actor):.3f} to 0.0"
            )
            ocr_gpus_per_actor = 0.0

        if embed_invoke_url and float(embed_gpus_per_actor) != 0.0:
            print(
                "[WARN] --embed-invoke-url is set; forcing --embed-gpus-per-actor from "
                f"{float(embed_gpus_per_actor):.3f} to 0.0"
            )
            embed_gpus_per_actor = 0.0

        input_path = Path(input_path)
        if input_path.is_file():
            file_patterns = [str(input_path)]
        elif input_path.is_dir():
            ext_map = {
                "txt": ["*.txt"],
                "html": ["*.html"],
                "doc": ["*.docx", "*.pptx"],
            }
            exts = ext_map.get(input_type, ["*.pdf"])
            file_patterns = [str(input_path / e) for e in exts]
        else:
            raise typer.BadParameter(f"Path does not exist: {input_path}")

        ingestor = create_ingestor(
            run_mode="batch",
            params=IngestorCreateParams(
                ray_address=ray_address, ray_log_to_driver=ray_log_to_driver, debug=bool(debug)
            ),
        )

        if input_type == "txt":
            ingestor = (
                ingestor.files(file_patterns)
                .extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        embed_granularity=embed_granularity,
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
        elif input_type == "html":
            ingestor = (
                ingestor.files(file_patterns)
                .extract_html(TextChunkParams(max_tokens=512, overlap_tokens=0))
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        embed_granularity=embed_granularity,
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
        elif input_type == "doc":
            ingestor = (
                ingestor.files(file_patterns)
                .extract(
                    ExtractParams(
                        extract_text=True,
                        extract_tables=True,
                        extract_charts=True,
                        extract_infographics=False,
                        use_graphic_elements=use_graphic_elements,
                        graphic_elements_invoke_url=graphic_elements_invoke_url,
                        inference_batch_size=page_elements_batch_size,
                        use_table_structure=use_table_structure,
                        table_output_format=table_output_format,
                        table_structure_invoke_url=table_structure_invoke_url,
                        page_elements_invoke_url=page_elements_invoke_url,
                        ocr_invoke_url=ocr_invoke_url,
                        batch_tuning={
                            "debug_run_id": str(runtime_metrics_prefix or "unknown"),
                            "pdf_extract_workers": pdf_extract_tasks,
                            "pdf_extract_num_cpus": pdf_extract_cpus_per_task,
                            "pdf_split_batch_size": pdf_split_batch_size,
                            "pdf_extract_batch_size": pdf_extract_batch_size,
                            "page_elements_batch_size": page_elements_batch_size,
                            "page_elements_workers": page_elements_actors,
                            "detect_workers": ocr_actors,
                            "detect_batch_size": ocr_batch_size,
                            "ocr_inference_batch_size": ocr_batch_size,
                            "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
                            "ocr_cpus_per_actor": ocr_cpus_per_actor,
                            "gpu_page_elements": page_elements_gpus_per_actor,
                            "gpu_ocr": ocr_gpus_per_actor,
                            "gpu_embed": embed_gpus_per_actor,
                            "nemotron_parse_workers": nemotron_parse_actors,
                            "gpu_nemotron_parse": nemotron_parse_gpus_per_actor,
                            "nemotron_parse_batch_size": nemotron_parse_batch_size,
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        batch_tuning={
                            "embed_workers": embed_actors,
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )
        else:
            ingestor = (
                ingestor.files(file_patterns)
                .extract(
                    ExtractParams(
                        extract_text=True,
                        extract_tables=True,
                        extract_charts=True,
                        extract_infographics=False,
                        use_graphic_elements=use_graphic_elements,
                        graphic_elements_invoke_url=graphic_elements_invoke_url,
                        inference_batch_size=page_elements_batch_size,
                        use_table_structure=use_table_structure,
                        table_output_format=table_output_format,
                        table_structure_invoke_url=table_structure_invoke_url,
                        page_elements_invoke_url=page_elements_invoke_url,
                        ocr_invoke_url=ocr_invoke_url,
                        batch_tuning={
                            "debug_run_id": str(runtime_metrics_prefix or "unknown"),
                            "pdf_extract_workers": pdf_extract_tasks,
                            "pdf_extract_num_cpus": pdf_extract_cpus_per_task,
                            "pdf_split_batch_size": pdf_split_batch_size,
                            "pdf_extract_batch_size": pdf_extract_batch_size,
                            "page_elements_batch_size": page_elements_batch_size,
                            "inference_batch_size": page_elements_batch_size,
                            "page_elements_workers": page_elements_actors,
                            "detect_workers": ocr_actors,
                            "detect_batch_size": ocr_batch_size,
                            "ocr_inference_batch_size": ocr_batch_size,
                            "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
                            "ocr_cpus_per_actor": ocr_cpus_per_actor,
                            "gpu_page_elements": page_elements_gpus_per_actor,
                            "gpu_ocr": ocr_gpus_per_actor,
                            "gpu_embed": embed_gpus_per_actor,
                            "nemotron_parse_workers": nemotron_parse_actors,
                            "gpu_nemotron_parse": nemotron_parse_gpus_per_actor,
                            "nemotron_parse_batch_size": nemotron_parse_batch_size,
                        },
                    )
                )
                .embed(
                    EmbedParams(
                        model_name=str(embed_model_name),
                        embed_invoke_url=embed_invoke_url,
                        embed_modality=embed_modality,
                        text_elements_modality=text_elements_modality,
                        structured_elements_modality=structured_elements_modality,
                        batch_tuning={
                            "embed_workers": embed_actors,
                            "embed_batch_size": int(embed_batch_size),
                            "embed_cpus_per_actor": float(embed_cpus_per_actor),
                        },
                    )
                )
                .vdb_upload(
                    VdbUploadParams(
                        lancedb={
                            "lancedb_uri": lancedb_uri,
                            "table_name": LANCEDB_TABLE,
                            "overwrite": True,
                            "create_index": True,
                            "hybrid": hybrid,
                        }
                    )
                )
            )

        logger.info("Running extraction...")
        ingest_start = time.perf_counter()

        ingest_results = (
            ingestor.ingest(
                params=IngestExecuteParams(
                    runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
                    runtime_metrics_prefix=runtime_metrics_prefix,
                )
            )
            .get_dataset()
            .materialize()
        )

        ingest_elapsed_s = time.perf_counter() - ingest_start
        num_rows = ingest_results.groupby("source_id").count().count()
        logger.info(
            f"Ingestion complete. {num_rows} rows procesed in "
            f"{ingest_elapsed_s:.2f} seconds. {num_rows/ingest_elapsed_s:.2f} PPS"
        )

        if isinstance(ingestor, BatchIngestor):
            error_rows = ingestor.get_error_rows(dataset=ingest_results).materialize()
            error_count = int(error_rows.count())

            # Error out, stop processing, and write top 5 errors rows to a local file for analysis.
            if error_count > 0:
                error_file = Path("ingest_errors.json").resolve()
                max_error_rows_to_write = 5
                error_rows_to_write = error_rows.take(min(max_error_rows_to_write, error_count))
                with error_file.open("w", encoding="utf-8") as fh:
                    json.dump(error_rows_to_write, fh, indent=2, default=str)
                    fh.write("\n")
                logger.error(
                    "Detected %d error row(s) in ingest results. Wrote first %d row(s) "
                    "to %s. Showing top 5 extracted errors and exiting before recall."
                    " Writing top(%d) error rows to %s",
                    error_count,
                    len(error_rows_to_write),
                    str(error_file),
                    int(max_error_rows_to_write),
                    str(error_file),
                )

                ray.shutdown()
                logger.error(f"Exiting with code 1 due to {error_count} error rows in ingest results.")
                raise typer.Exit(code=1)

        ray.shutdown()

        # ---------------------------------------------------------------------------
        # Recall calculation
        # ---------------------------------------------------------------------------
        query_csv = Path(query_csv)
        if not query_csv.exists():
            logger.warning(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
            return

        db = _lancedb().connect(lancedb_uri)
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
                logger.warning(f"LanceDB table {LANCEDB_TABLE!r} exists but is empty; skipping recall evaluation.")
                return
        except Exception:
            pass

        # Resolve the HF model ID for recall query embedding so aliases
        # (e.g. "nemo_retriever_v1") map to the correct model.
        from nemo_retriever.model import resolve_embed_model

        _recall_model = resolve_embed_model(str(embed_model_name))

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model=_recall_model,
            embedding_http_endpoint=embed_invoke_url,
            top_k=10,
            ks=(1, 5, 10),
            hybrid=hybrid,
            match_mode=recall_match_mode,
        )

        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)

        logger.info("\nRecall metrics (matching nemo_retriever.recall.core):")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
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
