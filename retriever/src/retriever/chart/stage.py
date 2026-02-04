from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)

import typer
from pathlib import Path

app = typer.Typer(help="Chart Extraction Stage")


def _validate_primitives_df(df: pd.DataFrame) -> None:
    if "metadata" not in df.columns:
        raise KeyError("Primitives DataFrame must include a 'metadata' column.")


@traceable_func(trace_name="retriever::chart_extraction")
def extract_chart_data_from_primitives_df(
    df_primitives: pd.DataFrame,
    *,
    extractor_config: ChartExtractorSchema,
    task_config: Optional[Dict[str, Any]] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Enrich chart primitives in-place by running YOLOX/OCR and writing chart content."""
    _validate_primitives_df(df_primitives)

    if task_config is None:
        task_config = {}

    execution_trace_log: Dict[str, Any] = {}
    try:
        out_df, info = extract_chart_data_from_image_internal(
            df_extraction_ledger=df_primitives,
            task_config=task_config,
            extraction_config=extractor_config,
            execution_trace_log=execution_trace_log,
        )
    except Exception:
        logger.exception("Chart extraction failed")
        raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)
    return out_df, info


@app.command("graphic-elements")
def render_graphic_elements(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config file. If set, values are loaded from YAML; explicitly passed CLI flags override YAML.",
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan recursively for *.pdf (can be provided via --config).",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF extraction method (e.g. 'pdfium', 'pdfium_hybrid', 'ocr', 'nemotron_parse', 'tika').",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help="Auth token for NIM-backed services (e.g. YOLOX / Nemotron Parse).",
    ),
    yolox_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--yolox-grpc-endpoint",
        help="YOLOX gRPC endpoint (e.g. 'page-elements:8001'). Required for method 'pdfium' family.",
    ),
    yolox_http_endpoint: Optional[str] = typer.Option(
        None,
        "--yolox-http-endpoint",
        help="YOLOX HTTP endpoint (e.g. 'http://page-elements:8000/v1/infer'). Required for method 'pdfium' family.",
    ),
    nemotron_parse_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-grpc-endpoint",
        help="Nemotron Parse gRPC endpoint (required for method 'nemotron_parse').",
    ),
    nemotron_parse_http_endpoint: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-http-endpoint",
        help="Nemotron Parse HTTP endpoint (required for method 'nemotron_parse').",
    ),
    nemotron_parse_model_name: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-model-name",
        help="Nemotron Parse model name (optional; defaults to schema default).",
    ),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text", help="Extract text primitives."),
    extract_images: bool = typer.Option(
        False, "--extract-images/--no-extract-images", help="Extract image primitives."
    ),
    extract_tables: bool = typer.Option(
        False, "--extract-tables/--no-extract-tables", help="Extract table primitives."
    ),
    extract_charts: bool = typer.Option(
        False, "--extract-charts/--no-extract-charts", help="Extract chart primitives."
    ),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", help="Extract infographic primitives."
    ),
    extract_page_as_image: bool = typer.Option(
        False, "--extract-page-as-image/--no-extract-page-as-image", help="Extract full page images as primitives."
    ),
    text_depth: str = typer.Option(
        "page",
        "--text-depth",
        help="Text depth for extracted text primitives: 'page' or 'document'.",
    ),
    write_json_outputs: bool = typer.Option(
        True,
        "--write-json-outputs/--no-write-json-outputs",
        help="Write one <pdf>.pdf_extraction.json sidecar per input PDF.",
    ),
    json_output_dir: Optional[Path] = typer.Option(
        None,
        "--json-output-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional directory to write JSON outputs into (instead of next to PDFs).",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of PDFs processed."),
) -> None:
    pass
