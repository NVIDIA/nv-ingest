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
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, TextIO

from nemo_retriever.utils.detection_summary import print_run_summary
import ray
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.ingest_modes.lancedb_utils import lancedb_schema
from nemo_retriever.model import resolve_embed_model
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vector_store.lancedb_store import handle_lancedb

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


def _ensure_lancedb_table(uri: str, table_name: str) -> None:
    """Ensure the local LanceDB URI exists and table can be opened.

    Creates an empty table with the expected schema if it does not exist yet.
    """
    Path(uri).mkdir(parents=True, exist_ok=True)

    db = _lancedb().connect(uri)
    try:
        db.open_table(table_name)
        return
    except Exception:
        pass

    import pyarrow as pa  # type: ignore

    schema = lancedb_schema()
    empty = pa.table({f.name: [] for f in schema}, schema=schema)
    db.create_table(table_name, data=empty, schema=schema, mode="create")


def _gold_to_doc_page(golden_key: str) -> tuple[str, str]:
    s = str(golden_key)
    if "_" not in s:
        return s, ""
    doc, page = s.rsplit("_", 1)
    return doc, page


def _hit_key_and_distance(hit: dict) -> tuple[str | None, float | None]:

    source_id = hit.get("source_id")
    page_number = hit.get("page_number")
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
    dpi: int = typer.Option(
        300,
        "--dpi",
        min=72,
        help="Render DPI for PDF page images (default: 300).",
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
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Optional bearer token for remote NIM endpoints. Defaults to NVIDIA_API_KEY, then NGC_API_KEY.",
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
        help="Input format: 'pdf', 'txt', 'html', 'doc', or 'image'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice), 'image' for standalone image files (PNG, JPEG, BMP, TIFF, SVG).",  # noqa: E501
    ),
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF text extraction method: 'pdfium' (native only), 'pdfium_hybrid' (native + OCR for scanned), 'ocr' (OCR all pages), or 'nemotron_parse' (Nemotron Parse only, auto-configured).",  # noqa: E501
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
    reranker: Optional[bool] = typer.Option(
        False, "--reranker/--no-reranker", help="Enable a re-ranking stage with a cross-encoder model."
    ),
    reranker_model_name: str = typer.Option(
        "nvidia/llama-nemotron-rerank-1b-v2",
        "--reranker-model-name",
        help="Cross-encoder model name for re-ranking stage (passed to .embed()).",
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
    text_chunk: bool = typer.Option(
        False,
        "--text-chunk",
        help=(
            "Re-chunk extracted page text by token count before embedding. "
            "Uses --text-chunk-max-tokens and --text-chunk-overlap-tokens (defaults: 1024, 150)."
        ),
    ),
    text_chunk_max_tokens: Optional[int] = typer.Option(
        None,
        "--text-chunk-max-tokens",
        help="Max tokens per text chunk (default: 1024). Implies --text-chunk.",
    ),
    text_chunk_overlap_tokens: Optional[int] = typer.Option(
        None,
        "--text-chunk-overlap-tokens",
        help="Token overlap between consecutive text chunks (default: 150). Implies --text-chunk.",
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
        remote_api_key = resolve_remote_api_key(api_key)
        extract_remote_api_key = (
            remote_api_key
            if any(
                (
                    page_elements_invoke_url,
                    ocr_invoke_url,
                    graphic_elements_invoke_url,
                    table_structure_invoke_url,
                )
            )
            else None
        )
        embed_remote_api_key = remote_api_key if embed_invoke_url else None

        if (
            any(
                (
                    page_elements_invoke_url,
                    ocr_invoke_url,
                    graphic_elements_invoke_url,
                    table_structure_invoke_url,
                    embed_invoke_url,
                )
            )
            and remote_api_key is None
        ):
            logger.warning(
                "Remote endpoint URL(s) were configured without an API key. "
                "If these endpoints are hosted on build.nvidia.com, set --api-key or NVIDIA_API_KEY."
            )

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

        # Map single-extension image shortcuts (e.g. "png", "jpg") to a specific
        # glob while routing through the "image" pipeline.
        _image_ext_map = {
            "png": ["*.png"],
            "jpg": ["*.jpg", "*.jpeg"],
            "jpeg": ["*.jpg", "*.jpeg"],
            "bmp": ["*.bmp"],
            "tiff": ["*.tiff", "*.tif"],
            "tif": ["*.tiff", "*.tif"],
            "svg": ["*.svg"],
        }
        # Remember the original input_type for glob selection before normalizing
        # to the canonical pipeline name.
        _original_input_type = input_type
        if input_type in _image_ext_map:
            input_type = "image"

        input_path = Path(input_path)
        if input_path.is_file():
            file_patterns = [str(input_path)]
        elif input_path.is_dir():
            ext_map = {
                "txt": ["*.txt"],
                "html": ["*.html"],
                "doc": ["*.docx", "*.pptx"],
                "image": ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif", "*.svg"],
            }
            # If a specific image extension was requested, use only that extension's globs.
            if _original_input_type in _image_ext_map:
                exts = _image_ext_map[_original_input_type]
            else:
                exts = ext_map.get(input_type, ["*.pdf"])
            import glob as _glob

            all_candidates = [str(input_path / e) for e in exts]
            # Only keep globs that match at least one file (avoids errors for
            # missing extensions when input_type covers multiple formats).
            file_patterns = [p for p in all_candidates if _glob.glob(p)]
            if not file_patterns:
                raise typer.BadParameter(
                    f"No files found for input_type={input_type!r} in {input_path} " f"(tried: {', '.join(exts)})"
                )
        else:
            raise typer.BadParameter(f"Path does not exist: {input_path}")

        ingestor = create_ingestor(
            run_mode="batch",
            params=IngestorCreateParams(
                ray_address=ray_address, ray_log_to_driver=ray_log_to_driver, debug=bool(debug)
            ),
        )

        # -- Shared params used by multiple input-type branches ----------------
        embed_params = EmbedParams(
            model_name=str(embed_model_name),
            embed_invoke_url=embed_invoke_url,
            api_key=embed_remote_api_key,
            embed_modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
            embed_granularity=embed_granularity,
            batch_tuning={
                "embed_workers": embed_actors,
                "embed_batch_size": int(embed_batch_size),
                "embed_cpus_per_actor": float(embed_cpus_per_actor),
                "gpu_embed": float(embed_gpus_per_actor),
            },
        )
        # txt/html don't use embed_granularity from batch_tuning the same way,
        # but the extra keys are harmlessly ignored by EmbedParams.

        # Detection batch_tuning shared by pdf, doc, and image pipelines.
        _detection_batch_tuning = {
            "debug_run_id": str(runtime_metrics_prefix or "unknown"),
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
        }

        # PDF-specific tuning keys (split/extract workers) on top of detection tuning.
        _pdf_batch_tuning = {
            **_detection_batch_tuning,
            "pdf_extract_workers": pdf_extract_tasks,
            "pdf_extract_num_cpus": pdf_extract_cpus_per_task,
            "pdf_split_batch_size": pdf_split_batch_size,
            "pdf_extract_batch_size": pdf_extract_batch_size,
        }

        # ExtractParams shared by detection-based pipelines (pdf, doc, image).
        def _extract_params(batch_tuning: dict, **overrides: Any) -> ExtractParams:
            return ExtractParams(
                method=method,
                dpi=int(dpi),
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
                api_key=extract_remote_api_key,
                use_graphic_elements=use_graphic_elements,
                graphic_elements_invoke_url=graphic_elements_invoke_url,
                inference_batch_size=page_elements_batch_size,
                use_table_structure=use_table_structure,
                table_output_format=table_output_format,
                table_structure_invoke_url=table_structure_invoke_url,
                page_elements_invoke_url=page_elements_invoke_url,
                ocr_invoke_url=ocr_invoke_url,
                batch_tuning={**batch_tuning, **overrides},
            )

        _text_chunk_params = TextChunkParams(
            max_tokens=text_chunk_max_tokens or 1024,
            overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
        )

        if input_type == "txt":
            ingestor = ingestor.files(file_patterns).extract_txt(_text_chunk_params)
        elif input_type == "html":
            ingestor = ingestor.files(file_patterns).extract_html(_text_chunk_params)
        elif input_type == "image":
            ingestor = ingestor.files(file_patterns).extract_image_files(_extract_params(_detection_batch_tuning))
        elif input_type == "doc":
            ingestor = ingestor.files(file_patterns).extract(_extract_params(_pdf_batch_tuning))
        else:
            ingestor = ingestor.files(file_patterns).extract(
                _extract_params(_pdf_batch_tuning, inference_batch_size=page_elements_batch_size)
            )

        enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
        if enable_text_chunk:
            ingestor = ingestor.split(_text_chunk_params)

        ingestor = ingestor.embed(embed_params)

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

        ingestion_only_total_time = time.perf_counter() - ingest_start

        # Capture the time it takes to download the Ray dataset to the local machine for reporting.
        ray_dataset_download_start = time.perf_counter()
        ingest_local_results = ingest_results.take_all()
        ray_dataset_download_time = time.perf_counter() - ray_dataset_download_start

        # Write to lancedb and capture the time it takes.
        lancedb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, LANCEDB_TABLE, hybrid=hybrid, mode="overwrite")
        lancedb_write_time = time.perf_counter() - lancedb_write_start

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

        _recall_model = resolve_embed_model(str(embed_model_name))

        cfg = RecallConfig(
            lancedb_uri=str(lancedb_uri),
            lancedb_table=str(LANCEDB_TABLE),
            embedding_model=_recall_model,
            embedding_http_endpoint=embed_invoke_url,
            embedding_api_key=embed_remote_api_key or "",
            top_k=10,
            ks=(1, 5, 10),
            hybrid=hybrid,
            match_mode=recall_match_mode,
            reranker=reranker_model_name if reranker else None,
        )

        # Capture recall only times.
        recall_start = time.perf_counter()
        _df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=cfg)
        recall_total_time = time.perf_counter() - recall_start

        total_time = time.perf_counter() - ingest_start

        # This processing has nothing to do with processing or performance so we exclude
        # it from the runtimes. Just getting row counts for metrics ...
        num_rows = ingest_results.groupby("source_id").count().count()

        ray.shutdown()

        # Print runtimes for easy user viewing at end
        print_run_summary(
            num_rows,
            input_path,
            hybrid,
            lancedb_uri,
            LANCEDB_TABLE,
            total_time,
            ingestion_only_total_time,
            ray_dataset_download_time,
            lancedb_write_time,
            recall_total_time,
            metrics,
        )

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
