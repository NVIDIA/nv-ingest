# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-process ingestion pipeline (no Ray) with optional recall evaluation.
Run with: uv run python -m nemo_retriever.examples.inprocess_pipeline <input-dir>
"""

import json
import time
from pathlib import Path
from typing import Optional

import lancedb
import typer
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score

app = typer.Typer()

LANCEDB_URI = "lancedb"
LANCEDB_TABLE = "nv-ingest"


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


def _print_pages_per_second(processed_pages: Optional[int], ingest_elapsed_s: float) -> None:
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print("Pages/sec: unavailable (could not estimate processed pages). " f"Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec (ingest only): {pps:.2f}")


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
    dist = float(hit.get("_distance")) if "_distance" in hit else None
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
        "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "--embed-model-name",
        help="Embedding model name passed to .embed().",
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
) -> None:
    _ = input_type

    if gpu_devices is not None and num_gpus is not None:
        raise typer.BadParameter("--gpu-devices and --num-gpus are mutually exclusive.")
    if gpu_devices is not None:
        gpu_device_list = [d.strip() for d in gpu_devices.split(",") if d.strip()]
    elif num_gpus is not None:
        gpu_device_list = [str(i) for i in range(num_gpus)] if num_gpus > 0 else ["0"]
    else:
        gpu_device_list = ["0"]

    input_dir = Path(input_dir)
    if input_type == "txt":
        glob_pattern = str(input_dir / "*.txt")
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract_txt(TextChunkParams(max_tokens=512, overlap_tokens=0))
            .embed(
                EmbedParams(
                    model_name=str(embed_model_name),
                    embed_invoke_url=embed_invoke_url,
                    embed_modality=embed_modality,
                    text_elements_modality=text_elements_modality,
                    structured_elements_modality=structured_elements_modality,
                )
            )
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": LANCEDB_URI,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                )
            )
        )
    elif input_type == "html":
        glob_pattern = str(input_dir / "*.html")
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract_html(TextChunkParams(max_tokens=512, overlap_tokens=0))
            .embed(
                EmbedParams(
                    model_name=str(embed_model_name),
                    embed_invoke_url=embed_invoke_url,
                    embed_modality=embed_modality,
                    text_elements_modality=text_elements_modality,
                    structured_elements_modality=structured_elements_modality,
                )
            )
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": LANCEDB_URI,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                )
            )
        )
    elif input_type == "doc":
        # DOCX/PPTX: same pipeline as PDF; inprocess loader converts to PDF then splits.
        doc_globs = [str(input_dir / "*.docx"), str(input_dir / "*.pptx")]
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(doc_globs)
            .extract(
                ExtractParams(
                    method="pdfium",
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    page_elements_invoke_url=page_elements_invoke_url,
                    ocr_invoke_url=ocr_invoke_url,
                )
            )
            .embed(
                EmbedParams(
                    model_name=str(embed_model_name),
                    embed_invoke_url=embed_invoke_url,
                    embed_modality=embed_modality,
                    text_elements_modality=text_elements_modality,
                    structured_elements_modality=structured_elements_modality,
                )
            )
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": LANCEDB_URI,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                )
            )
        )
    else:
        glob_pattern = str(input_dir / "*.pdf")
        ingestor = create_ingestor(run_mode="inprocess")
        ingestor = (
            ingestor.files(glob_pattern)
            .extract(
                ExtractParams(
                    method="pdfium",
                    extract_text=True,
                    extract_tables=True,
                    extract_charts=True,
                    extract_infographics=False,
                    page_elements_invoke_url=page_elements_invoke_url,
                    ocr_invoke_url=ocr_invoke_url,
                )
            )
            .embed(
                EmbedParams(
                    model_name=str(embed_model_name),
                    embed_invoke_url=embed_invoke_url,
                    embed_modality=embed_modality,
                    text_elements_modality=text_elements_modality,
                    structured_elements_modality=structured_elements_modality,
                )
            )
            .vdb_upload(
                VdbUploadParams(
                    lancedb={
                        "lancedb_uri": LANCEDB_URI,
                        "table_name": LANCEDB_TABLE,
                        "overwrite": True,
                        "create_index": True,
                    }
                )
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
    processed_pages = _estimate_processed_pages(LANCEDB_URI, LANCEDB_TABLE)
    print("Extraction complete.")

    # ---------------------------------------------------------------------------
    # Recall calculation (optional)
    # ---------------------------------------------------------------------------
    query_csv = Path(query_csv)
    if not query_csv.exists():
        print(f"Query CSV not found at {query_csv}; skipping recall evaluation.")
        _print_pages_per_second(processed_pages, ingest_elapsed_s)
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
        doc, page = _gold_to_doc_page(g)

        scored_hits: list[tuple[str, float | None]] = []
        for h in hits:
            key, dist = _hit_key_and_distance(h)
            if key:
                scored_hits.append((key, dist))

        top_keys = [k for (k, _d) in scored_hits]
        hit = _is_hit_at_k(g, top_keys, cfg.top_k)

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
    _print_pages_per_second(processed_pages, ingest_elapsed_s)


if __name__ == "__main__":
    app()
