# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-process ingestion pipeline CLI.

Run with: nr inprocess <input-path>
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(help="Run the in-process ingestion pipeline (no Ray required).")

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable debug-level logging.",
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', or 'doc'. "
        "Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), "
        "'doc' for .docx/.pptx (converted to PDF via LibreOffice).",
    ),
    # -- Extract params --------------------------------------------------------
    extract_text: bool = typer.Option(
        True,
        "--extract-text/--no-extract-text",
        help="Extract text from PDF pages.",
    ),
    extract_tables: bool = typer.Option(
        True,
        "--extract-tables/--no-extract-tables",
        help="Extract tables from PDF pages.",
    ),
    extract_charts: bool = typer.Option(
        True,
        "--extract-charts/--no-extract-charts",
        help="Extract charts from PDF pages.",
    ),
    extract_infographics: bool = typer.Option(
        False,
        "--extract-infographics/--no-extract-infographics",
        help="Extract infographics from PDF pages.",
    ),
    use_table_structure: bool = typer.Option(
        False,
        "--use-table-structure",
        help="Enable the combined table-structure + OCR stage for tables.",
    ),
    table_output_format: Optional[str] = typer.Option(
        None,
        "--table-output-format",
        help="Table output format: 'pseudo_markdown' (OCR-only) or 'markdown' (table-structure + OCR).",
    ),
    inference_batch_size: Optional[int] = typer.Option(
        None,
        "--inference-batch-size",
        help="Inference batch size for detection models.",
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
    table_structure_invoke_url: Optional[str] = typer.Option(
        None,
        "--table-structure-invoke-url",
        help="Optional remote endpoint URL for table-structure model inference.",
    ),
    # -- Embed params ----------------------------------------------------------
    embed_model_name: str = typer.Option(
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
    ),
    embed_invoke_url: Optional[str] = typer.Option(
        None,
        "--embed-invoke-url",
        help="Optional remote endpoint URL for embedding model inference.",
    ),
    embed_modality: str = typer.Option(
        "text",
        "--embed-modality",
        help="Default embedding modality: 'text', 'image', or 'text_image'.",
    ),
    embed_granularity: str = typer.Option(
        "element",
        "--embed-granularity",
        help="Embedding granularity: 'element' (per table/chart/text) or 'page' (per page).",
    ),
    text_elements_modality: Optional[str] = typer.Option(
        None,
        "--text-elements-modality",
        help="Embedding modality override for page-text rows. Falls back to --embed-modality.",
    ),
    structured_elements_modality: Optional[str] = typer.Option(
        None,
        "--structured-elements-modality",
        help="Embedding modality override for table/chart/infographic rows. Falls back to --embed-modality.",
    ),
    # -- VDB upload params -----------------------------------------------------
    lancedb_uri: str = typer.Option(
        LANCEDB_URI,
        "--lancedb-uri",
        help="LanceDB URI/path for this run.",
    ),
    lancedb_table: str = typer.Option(
        LANCEDB_TABLE,
        "--lancedb-table",
        help="LanceDB table name.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
    ),
    overwrite: bool = typer.Option(
        True,
        "--overwrite/--no-overwrite",
        help="Overwrite existing LanceDB table (False to append).",
    ),
    # -- Output ----------------------------------------------------------------
    output_directory: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Optional directory to write per-document JSON results.",
    ),
    # -- Execution params ------------------------------------------------------
    parallel: bool = typer.Option(
        False,
        "--parallel/--no-parallel",
        help="Enable parallel CPU extraction via ProcessPoolExecutor.",
    ),
    max_workers: Optional[int] = typer.Option(
        None,
        "--max-workers",
        help="Max workers for parallel CPU extraction (default: CPU count).",
    ),
    gpu_devices: Optional[str] = typer.Option(
        None,
        "--gpu-devices",
        help="Comma-separated GPU device IDs for multi-GPU pipelined execution (e.g. '0,1').",
    ),
    page_chunk_size: int = typer.Option(
        32,
        "--page-chunk-size",
        help="Number of pages per chunk for parallel processing.",
    ),
    show_progress: bool = typer.Option(
        True,
        "--show-progress/--no-show-progress",
        help="Show progress bars during ingestion.",
    ),
) -> None:
    """Run the in-process ingestion pipeline on local documents (no Ray required)."""
    from nemo_retriever.ingest_modes.inprocess import InProcessIngestor
    from nemo_retriever.params import (
        EmbedParams,
        ExtractParams,
        HtmlChunkParams,
        IngestExecuteParams,
        TextChunkParams,
        VdbUploadParams,
    )

    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    lancedb_uri_abs = str(Path(lancedb_uri).expanduser().resolve())

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

    ingestor = InProcessIngestor()
    ingestor = ingestor.files(file_patterns)

    if input_type == "txt":
        ingestor = ingestor.extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
    elif input_type == "html":
        ingestor = ingestor.extract_html(HtmlChunkParams(max_tokens=512, overlap_tokens=0))
    else:
        extract_kw: dict = dict(
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            use_table_structure=use_table_structure,
            table_output_format=table_output_format,
        )
        if inference_batch_size is not None:
            extract_kw["inference_batch_size"] = inference_batch_size
        if page_elements_invoke_url:
            extract_kw["page_elements_invoke_url"] = page_elements_invoke_url
        if ocr_invoke_url:
            extract_kw["ocr_invoke_url"] = ocr_invoke_url
        if table_structure_invoke_url:
            extract_kw["table_structure_invoke_url"] = table_structure_invoke_url
        ingestor = ingestor.extract(ExtractParams(**extract_kw))

    ingestor = ingestor.embed(
        EmbedParams(
            model_name=str(embed_model_name),
            embed_invoke_url=embed_invoke_url,
            embed_modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
            embed_granularity=embed_granularity,
        )
    )

    ingestor = ingestor.vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": lancedb_uri_abs,
                "table_name": lancedb_table,
                "overwrite": overwrite,
                "create_index": True,
                "hybrid": hybrid,
            }
        )
    )

    if output_directory:
        ingestor = ingestor.save_to_disk(output_directory=output_directory)

    parsed_gpu_devices = None
    if gpu_devices:
        parsed_gpu_devices = [int(d.strip()) for d in gpu_devices.split(",") if d.strip()]

    logger.info("Running in-process ingestion...")
    start = time.perf_counter()

    ingestor.ingest(
        params=IngestExecuteParams(
            show_progress=show_progress,
            parallel=parallel,
            max_workers=max_workers,
            gpu_devices=parsed_gpu_devices,
            page_chunk_size=page_chunk_size,
        )
    )

    elapsed = time.perf_counter() - start
    logger.info(f"In-process ingestion complete in {elapsed:.2f}s")


if __name__ == "__main__":
    app()
