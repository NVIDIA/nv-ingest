from __future__ import annotations

from pathlib import Path

import ray
import typer

from .pipeline import run_full_pipeline, run_pdf_extraction
from retriever.chart.config import ChartExtractionStageConfig
from retriever.infographic.config import InfographicExtractionStageConfig
from retriever.pdf.config import PDFExtractionStageConfig
from retriever.table.config import TableExtractionStageConfig
from retriever.text_embed.config import TextEmbeddingStageConfig
from retriever.vector_store.lancedb_store import LanceDBConfig
from retriever.recall import RecallConfig, evaluate_recall

app = typer.Typer(no_args_is_help=True, help="Retriever skeleton CLI (Ray Data + PDF + pandas)")


@app.command("pdf-to-csv")
def pdf_to_csv(
    input_dir: Path = typer.Argument(..., help="Directory to scan recursively for *.pdf"),
    out_csv: Path = typer.Argument(..., help="Output CSV path"),
    runner: str = typer.Option(
        "python",
        help="Execution mode: 'python' (no Ray required) or 'ray-data' (Ray Data map_batches).",
    ),
    method: str = typer.Option(
        "pdfium",
        help="PDF extraction method (e.g. 'pdfium' or 'nemotron_parse').",
    ),
    auth_token: str | None = typer.Option(
        None,
        help="Auth token for NIM-backed services (e.g. YOLOX / Nemotron Parse).",
    ),
    yolox_grpc_endpoint: str | None = typer.Option(
        None,
        help="YOLOX gRPC endpoint (e.g. 'page-elements:8001').",
    ),
    yolox_http_endpoint: str | None = typer.Option(
        None,
        help="YOLOX HTTP endpoint (e.g. 'http://page-elements:8000/v1/infer').",
    ),
    nemotron_parse_grpc_endpoint: str | None = typer.Option(
        None,
        help="Nemotron Parse gRPC endpoint.",
    ),
    nemotron_parse_http_endpoint: str | None = typer.Option(
        None,
        help="Nemotron Parse HTTP endpoint (e.g. 'http://nemotron-parse:8000/v1/chat/completions').",
    ),
    nemotron_parse_model_name: str | None = typer.Option(
        None,
        help="Nemotron Parse model name (default comes from nv-ingest-api schema).",
    ),
    batch_size: int = typer.Option(16, help="Ray Data batch size when runner='ray-data'."),
    extract_text: bool = typer.Option(True, help="Extract text primitives."),
    extract_images: bool = typer.Option(False, help="Extract image primitives."),
    extract_tables: bool = typer.Option(False, help="Extract table primitives."),
    extract_charts: bool = typer.Option(False, help="Extract chart primitives."),
    extract_infographics: bool = typer.Option(False, help="Extract infographic primitives."),
) -> None:
    """Extract PDF primitives using `nv-ingest-api` and write a CSV via pandas."""
    pdfs = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    if not pdfs:
        raise typer.Exit(code=2)

    needs_yolox = (
        extract_images
        or extract_tables
        or extract_charts
        or extract_infographics
        or method in {"pdfium_hybrid", "ocr"}
    )
    if needs_yolox and not (yolox_grpc_endpoint or yolox_http_endpoint):
        raise typer.BadParameter(
            "YOLOX endpoint required for the requested extractions. "
            "Set --yolox-grpc-endpoint or --yolox-http-endpoint, or disable structured/image extraction flags."
        )

    if method == "nemotron_parse" and not (nemotron_parse_grpc_endpoint or nemotron_parse_http_endpoint):
        raise typer.BadParameter(
            "Nemotron Parse endpoint required for method 'nemotron_parse'. "
            "Set --nemotron-parse-grpc-endpoint or --nemotron-parse-http-endpoint."
        )

    extractor_cfg: dict = {}
    # Only include sub-configs when we have enough information for schema validation.
    if yolox_grpc_endpoint or yolox_http_endpoint:
        extractor_cfg["pdfium_config"] = {
            "auth_token": auth_token,
            "yolox_endpoints": [yolox_grpc_endpoint, yolox_http_endpoint],
        }

    if nemotron_parse_grpc_endpoint or nemotron_parse_http_endpoint:
        extractor_cfg["nemotron_parse_config"] = {
            "auth_token": auth_token,
            "yolox_endpoints": [yolox_grpc_endpoint, yolox_http_endpoint],
            "nemotron_parse_endpoints": [nemotron_parse_grpc_endpoint, nemotron_parse_http_endpoint],
            "nemotron_parse_model_name": nemotron_parse_model_name,
        }

    task_cfg: dict = {
        "method": method,
        "params": {
            "extract_text": extract_text,
            "extract_images": extract_images,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
        },
    }

    if runner == "ray-data":
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            result = run_pdf_extraction(
                pdfs,
                runner="ray-data",
                stage_cfg=PDFExtractionStageConfig(batch_size=batch_size),
                extractor_cfg_dict=extractor_cfg,
                task_cfg=task_cfg,
            )
        finally:
            ray.shutdown()
    else:
        result = run_pdf_extraction(
            pdfs,
            runner="python",
            extractor_cfg_dict=extractor_cfg,
            task_cfg=task_cfg,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.extracted_df.to_csv(out_csv, index=False)


@app.command("pipeline-to-csv")
def pipeline_to_csv(
    input_dir: Path = typer.Argument(..., help="Directory to scan recursively for *.pdf"),
    out_csv: Path = typer.Argument(..., help="Output CSV path"),
    runner: str = typer.Option(
        "python",
        help="Execution mode: 'python' (no Ray required) or 'ray-data' (Ray Data map_batches).",
    ),
    auth_token: str | None = typer.Option(None, help="Auth token for NIM-backed services (YOLOX / OCR)."),
    page_elements_grpc_endpoint: str | None = typer.Option(None, help="YOLOX page-elements gRPC endpoint."),
    page_elements_http_endpoint: str | None = typer.Option(None, help="YOLOX page-elements HTTP endpoint."),
    graphic_elements_grpc_endpoint: str | None = typer.Option(None, help="YOLOX graphic-elements gRPC endpoint."),
    graphic_elements_http_endpoint: str | None = typer.Option(None, help="YOLOX graphic-elements HTTP endpoint."),
    table_structure_grpc_endpoint: str | None = typer.Option(None, help="YOLOX table-structure gRPC endpoint."),
    table_structure_http_endpoint: str | None = typer.Option(None, help="YOLOX table-structure HTTP endpoint."),
    ocr_grpc_endpoint: str | None = typer.Option(None, help="OCR gRPC endpoint."),
    ocr_http_endpoint: str | None = typer.Option(None, help="OCR HTTP endpoint."),
    embedding_endpoint: str = typer.Option("http://embedding:8000/v1", help="Embedding NIM endpoint."),
    embedding_model: str = typer.Option("nvidia/llama-3.2-nv-embedqa-1b-v2", help="Embedding model name."),
    embedding_api_key: str | None = typer.Option(None, help="API key for embedding service (optional)."),
    batch_size_pdf: int = typer.Option(16, help="Ray Data batch size for PDF extraction."),
    batch_size_structured: int = typer.Option(64, help="Ray Data batch size for infographic/table/chart stages."),
    batch_size_embed: int = typer.Option(256, help="Ray Data batch size for embedding stage."),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show tqdm progress bars (python runner only).",
    ),
    lancedb_uri: str | None = typer.Option(
        None,
        help="If set, write embedding vectors to a LanceDB database at this URI (e.g. './lancedb').",
    ),
    lancedb_table: str = typer.Option("embeddings", help="LanceDB table name."),
    lancedb_mode: str = typer.Option("append", help="LanceDB write mode: append|overwrite."),
) -> None:
    """Run PDF -> infographic/table/chart -> embed as a single pipeline and write a CSV."""
    pdfs = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    if not pdfs:
        raise typer.Exit(code=2)

    if not (page_elements_grpc_endpoint or page_elements_http_endpoint):
        raise typer.BadParameter(
            "YOLOX page-elements endpoint required. "
            "Set --page-elements-grpc-endpoint or --page-elements-http-endpoint."
        )
    if not (graphic_elements_grpc_endpoint or graphic_elements_http_endpoint):
        raise typer.BadParameter(
            "YOLOX graphic-elements endpoint required. "
            "Set --graphic-elements-grpc-endpoint or --graphic-elements-http-endpoint."
        )
    if not (table_structure_grpc_endpoint or table_structure_http_endpoint):
        raise typer.BadParameter(
            "YOLOX table-structure endpoint required. "
            "Set --table-structure-grpc-endpoint or --table-structure-http-endpoint."
        )
    if not (ocr_grpc_endpoint or ocr_http_endpoint):
        raise typer.BadParameter("OCR endpoint required. Set --ocr-grpc-endpoint or --ocr-http-endpoint.")

    pdf_extractor_cfg = {
        "pdfium_config": {
            "auth_token": auth_token,
            # PDF extraction uses YOLOX page-elements.
            "yolox_endpoints": [page_elements_grpc_endpoint, page_elements_http_endpoint],
        }
    }
    pdf_task_cfg = {
        "method": "pdfium",
        "params": {
            "extract_text": True,
            "extract_images": True,
            "extract_tables": True,
            "extract_charts": True,
            "extract_infographics": True,
        },
    }

    infographic_cfg = {
        "endpoint_config": {
            "auth_token": auth_token,
            "ocr_endpoints": [ocr_grpc_endpoint, ocr_http_endpoint],
        }
    }
    table_cfg = {
        "endpoint_config": {
            "auth_token": auth_token,
            # Table extraction uses YOLOX table-structure.
            "yolox_endpoints": [table_structure_grpc_endpoint, table_structure_http_endpoint],
            "ocr_endpoints": [ocr_grpc_endpoint, ocr_http_endpoint],
        }
    }
    chart_cfg = {
        "endpoint_config": {
            "auth_token": auth_token,
            # Chart extraction uses YOLOX graphic-elements.
            "yolox_endpoints": [graphic_elements_grpc_endpoint, graphic_elements_http_endpoint],
            "ocr_endpoints": [ocr_grpc_endpoint, ocr_http_endpoint],
        }
    }
    embed_cfg = {
        "api_key": embedding_api_key,
        "embedding_model": embedding_model,
        "embedding_nim_endpoint": embedding_endpoint,
    }

    lancedb = None if not lancedb_uri else LanceDBConfig(uri=lancedb_uri, table=lancedb_table, mode=lancedb_mode)

    if runner == "ray-data":
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        try:
            result = run_full_pipeline(
                pdfs,
                runner="ray-data",
                pdf_stage_cfg=PDFExtractionStageConfig(batch_size=batch_size_pdf),
                infographic_stage_cfg=InfographicExtractionStageConfig(batch_size=batch_size_structured),
                table_stage_cfg=TableExtractionStageConfig(batch_size=batch_size_structured),
                chart_stage_cfg=ChartExtractionStageConfig(batch_size=batch_size_structured),
                embed_stage_cfg=TextEmbeddingStageConfig(batch_size=batch_size_embed),
                pdf_extractor_cfg_dict=pdf_extractor_cfg,
                infographic_extractor_cfg_dict=infographic_cfg,
                table_extractor_cfg_dict=table_cfg,
                chart_extractor_cfg_dict=chart_cfg,
                text_embedding_cfg_dict=embed_cfg,
                pdf_task_cfg=pdf_task_cfg,
                embed_task_cfg={},
                lancedb=lancedb,
            )
        finally:
            ray.shutdown()
    else:
        result = run_full_pipeline(
            pdfs,
            runner="python",
            pdf_extractor_cfg_dict=pdf_extractor_cfg,
            infographic_extractor_cfg_dict=infographic_cfg,
            table_extractor_cfg_dict=table_cfg,
            chart_extractor_cfg_dict=chart_cfg,
            text_embedding_cfg_dict=embed_cfg,
            pdf_task_cfg=pdf_task_cfg,
            embed_task_cfg={},
            progress=progress,
            lancedb=lancedb,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(out_csv, index=False)


@app.command("recall")
def recall(
    query_csv: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="CSV with columns: query,pdf,page (or gt_page)."),
    lancedb_uri: str = typer.Option("./lancedb", help="LanceDB URI (folder path or remote URI)."),
    lancedb_table: str = typer.Option("embeddings", help="LanceDB table name."),
    embedding_endpoint: str = typer.Option("http://embedding:8000/v1", help="Embedding NIM endpoint."),
    embedding_model: str = typer.Option("nvidia/llama-3.2-nv-embedqa-1b-v2", help="Embedding model name."),
    embedding_api_key: str | None = typer.Option(None, help="API key for embedding service (optional)."),
    top_k: int = typer.Option(10, help="Top-k retrieval depth."),
    output_dir: Path | None = typer.Option(
        Path("./outputs/recall_analysis"),
        help="If set, write per-query results CSV here.",
    ),
) -> None:
    """Evaluate recall@k by embedding queries and searching LanceDB vectors."""
    cfg = RecallConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedding_endpoint=embedding_endpoint,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key or "",
        top_k=top_k,
    )
    report = evaluate_recall(query_csv, cfg=cfg, output_dir=output_dir)
    metrics = report["metrics"]
    lines = [f"queries={report['n_queries']} top_k={report['top_k']}"]
    for k, v in metrics.items():
        lines.append(f"{k}={v:.4f}")
    if report.get("saved", {}).get("results_csv"):
        lines.append(f"results_csv={report['saved']['results_csv']}")
    typer.echo(" ".join(lines))


def main(argv: list[str] | None = None) -> int:
    # Keep a callable main() for console_scripts compatibility.
    app(prog_name="retriever", args=argv)
    return 0

