# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-process ingestion pipeline (no Ray) with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.inprocess_pipeline <input-dir>
"""

import time
from pathlib import Path
from typing import Optional

import lancedb
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.examples.common import estimate_processed_pages, print_pages_per_second
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.recall.core import (
    RecallConfig,
    gold_to_doc_page,
    hit_key_and_distance,
    is_hit_at_k,
    retrieve_and_score,
)

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


@app.command()
def main(
    input_path: Path = typer.Argument(
        ...,
        help="File or directory containing PDFs, .txt, .html, or .doc/.pptx files to ingest.",
        path_type=Path,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf', 'txt', 'html', 'doc', or 'image'. Use 'txt' for .txt, 'html' for .html (markitdown -> chunks), 'doc' for .docx/.pptx (converted to PDF via LibreOffice), 'image' for standalone image files (PNG, JPEG, BMP, TIFF, SVG).",  # noqa: E501
    ),
    query_csv: Path = typer.Option(
        "bo767_query_gt.csv",
        "--query-csv",
        path_type=Path,
        help="Path to query CSV for recall evaluation. Default: bo767_query_gt.csv (current directory). Recall is skipped if the file does not exist.",  # noqa: E501
    ),
    no_recall_details: bool = typer.Option(
        False,
        "--no-recall-details",
        help="Do not print per-query retrieval details (query, gold, hits). Only the missed-gold summary and recall metrics are printed.",  # noqa: E501
    ),
    max_workers: int = typer.Option(
        16,
        "--max-workers",
        help="Maximum number of parallel ingest workers.",
    ),
    gpu_devices: Optional[str] = typer.Option(
        None,
        "--gpu-devices",
        help="Comma-separated GPU device IDs (e.g. --gpu-devices 0,1,2). Mutually exclusive with --num-gpus.",
    ),
    num_gpus: Optional[int] = typer.Option(
        None,
        "--num-gpus",
        help="Number of GPUs to use, starting from device 0 (e.g. --num-gpus 2 → GPUs 0,1). Mutually exclusive with --gpu-devices.",  # noqa: E501
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
    embed_model_name: str = typer.Option(
        "nvidia/llama-nemotron-embed-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF text extraction method: 'pdfium' (native only), 'pdfium_hybrid' (native + OCR for scanned), 'ocr' (OCR all pages), or 'nemotron_parse' (Nemotron Parse only).",  # noqa: E501
    ),
    embed_modality: str = typer.Option(
        "text",
        "--embed-modality",
        help="Default embedding modality for all element types: "
        "'text', 'image', or 'text_image' ('image_text' is also accepted).",
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
    embed_granularity: str = typer.Option(
        "element",
        "--embed-granularity",
        help="Embedding granularity: 'element' (one row per table/chart/text) or 'page' (one row per page).",
    ),
    use_graphic_elements: bool = typer.Option(
        False,
        "--use-graphic-elements",
        help="Enable the combined graphic-elements + OCR stage for charts (requires extract_charts).",
    ),
    graphic_elements_invoke_url: Optional[str] = typer.Option(
        None,
        "--graphic-elements-invoke-url",
        help="Optional remote endpoint URL for graphic-elements model inference.",
    ),
    hybrid: bool = typer.Option(
        False,
        "--hybrid/--no-hybrid",
        help="Enable LanceDB hybrid mode (dense + FTS text).",
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
    if gpu_devices is not None and num_gpus is not None:
        raise typer.BadParameter("--gpu-devices and --num-gpus are mutually exclusive.")
    if gpu_devices is not None:
        gpu_device_list = [d.strip() for d in gpu_devices.split(",") if d.strip()]
    elif num_gpus is not None:
        gpu_device_list = [str(i) for i in range(num_gpus)] if num_gpus > 0 else ["0"]
    else:
        gpu_device_list = ["0"]

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
        exts = ext_map.get(input_type, ["*.pdf"])
        file_patterns = [str(input_path / e) for e in exts]
    else:
        raise typer.BadParameter(f"Path does not exist: {input_path}")

    ingestor = create_ingestor(run_mode="inprocess")
    if input_type == "txt":
        ingestor = ingestor.files(file_patterns).extract_txt(
            TextChunkParams(
                max_tokens=text_chunk_max_tokens or 1024,
                overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
            )
        )
    elif input_type == "html":
        ingestor = ingestor.files(file_patterns).extract_html(
            TextChunkParams(
                max_tokens=text_chunk_max_tokens or 1024,
                overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
            )
        )
    elif input_type == "image":
        ingestor = ingestor.files(file_patterns).extract_image_files(
            ExtractParams(
                method=method,
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
                use_graphic_elements=use_graphic_elements,
                graphic_elements_invoke_url=graphic_elements_invoke_url,
                use_table_structure=use_table_structure,
                table_output_format=table_output_format,
                table_structure_invoke_url=table_structure_invoke_url,
                page_elements_invoke_url=page_elements_invoke_url,
                ocr_invoke_url=ocr_invoke_url,
            )
        )
    elif input_type == "doc":
        ingestor = ingestor.files(file_patterns).extract(
            ExtractParams(
                method=method,
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
                use_graphic_elements=use_graphic_elements,
                graphic_elements_invoke_url=graphic_elements_invoke_url,
                use_table_structure=use_table_structure,
                table_output_format=table_output_format,
                table_structure_invoke_url=table_structure_invoke_url,
                page_elements_invoke_url=page_elements_invoke_url,
                ocr_invoke_url=ocr_invoke_url,
            )
        )
    else:
        ingestor = ingestor.files(file_patterns).extract(
            ExtractParams(
                method=method,
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_infographics=False,
                use_graphic_elements=use_graphic_elements,
                graphic_elements_invoke_url=graphic_elements_invoke_url,
                use_table_structure=use_table_structure,
                table_output_format=table_output_format,
                table_structure_invoke_url=table_structure_invoke_url,
                page_elements_invoke_url=page_elements_invoke_url,
                ocr_invoke_url=ocr_invoke_url,
            )
        )

    enable_text_chunk = text_chunk or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
    if enable_text_chunk:
        ingestor = ingestor.split(
            TextChunkParams(
                max_tokens=text_chunk_max_tokens or 1024,
                overlap_tokens=text_chunk_overlap_tokens if text_chunk_overlap_tokens is not None else 150,
            )
        )

    ingestor = ingestor.embed(
        EmbedParams(
            model_name=str(embed_model_name),
            embed_invoke_url=embed_invoke_url,
            embed_modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
            embed_granularity=embed_granularity,
        )
    ).vdb_upload(
        VdbUploadParams(
            lancedb={
                "lancedb_uri": LANCEDB_URI,
                "table_name": LANCEDB_TABLE,
                "overwrite": True,
                "create_index": True,
                "hybrid": hybrid,
            }
        )
    )

    print("Running extraction...")
    ingest_start = time.perf_counter()
    ingestor.ingest(
        params=IngestExecuteParams(
            parallel=True,
            max_workers=max_workers,
            gpu_devices=gpu_device_list,
            show_progress=True,
        )
    )
    ingest_elapsed_s = time.perf_counter() - ingest_start
    processed_pages = estimate_processed_pages(LANCEDB_URI, LANCEDB_TABLE)
    print("Extraction complete.")

    # ---------------------------------------------------------------------------
    # Recall calculation (optional)
    # ---------------------------------------------------------------------------
    query_csv = Path(query_csv)
    if not query_csv.exists():
        print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        print_pages_per_second(processed_pages, ingest_elapsed_s)
        return

    db = lancedb.connect(f"./{LANCEDB_URI}")
    table = db.open_table(LANCEDB_TABLE)
    unique_basenames = table.to_pandas()["pdf_basename"].unique()
    print(f"Unique basenames: {unique_basenames}")

    # Resolve the HF model ID for recall query embedding so aliases
    # (e.g. "nemo_retriever_v1") map to the correct model.
    from nemo_retriever.model import resolve_embed_model

    _recall_model = resolve_embed_model(str(embed_model_name))

    cfg = RecallConfig(
        lancedb_uri=str(LANCEDB_URI),
        lancedb_table=str(LANCEDB_TABLE),
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
    for i, (q, g, hits) in enumerate(
        zip(
            _df_query["query"].astype(str).tolist(),
            _gold,
            _raw_hits,
        )
    ):
        doc, page = gold_to_doc_page(g)

        scored_hits: list[tuple[str, float | None]] = []
        for h in hits:
            key, dist = hit_key_and_distance(h)
            if key:
                scored_hits.append((key, dist))

        top_keys = [k for (k, _d) in scored_hits]
        hit = is_hit_at_k(g, top_keys, cfg.top_k, match_mode="pdf_page")

        if not no_recall_details:
            ext = (
                ".txt"
                if input_type == "txt"
                else (".html" if input_type == "html" else (".docx" if input_type == "doc" else ".pdf"))
            )
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
            ext = (
                ".txt"
                if input_type == "txt"
                else (".html" if input_type == "html" else (".docx" if input_type == "doc" else ".pdf"))
            )
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
    print_pages_per_second(processed_pages, ingest_elapsed_s)


if __name__ == "__main__":
    app()
