"""
Batch ingestion pipeline with optional recall evaluation.
Run with: uv run python -m retriever.examples.batch_pipeline <input-dir>
"""
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import lancedb
import ray
import typer
from retriever import create_ingestor
from retriever.recall.core import RecallConfig, retrieve_and_score

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
        print(
            "Pages/sec: unavailable (could not estimate processed pages). "
            f"Ingest time: {ingest_elapsed_s:.2f}s"
        )
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
    dist = float(hit.get("_distance")) if "_distance" in hit else None
    return key, dist


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDFs or .txt files to ingest.",
        path_type=Path,
        exists=True,
    ),
    input_type: str = typer.Option(
        "pdf",
        "--input-type",
        help="Input format: 'pdf' or 'txt'. Use 'txt' for a directory of .txt files (tokenizer-based chunking).",
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
) -> None:
    os.environ.setdefault("NEMOTRON_OCR_MODEL_DIR", str(Path.cwd() / "nemotron-ocr-v1"))
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
        print(
            "[WARN] --ocr-invoke-url is set; forcing --gpu-ocr from "
            f"{float(gpu_ocr):.3f} to 0.0"
        )
        gpu_ocr = 0.0

    # Resolve Ray: start a head node, connect to given address, or run in-process
    if start_ray:
        subprocess.run(["ray", "start", "--head"], check=True, env=os.environ)
        ray_address = "auto"
    # else: use ray_address as-is (None → in-process, or URL to existing cluster)

    input_dir = Path(input_dir)
    if input_type == "txt":
        glob_pattern = str(input_dir / "*.txt")
        ingestor = create_ingestor(run_mode="batch", ray_address=ray_address)
        ingestor = (
            ingestor.files(glob_pattern)
            .extract_txt(max_tokens=512, overlap_tokens=0)
            .embed(model_name="nemo_retriever_v1")
            .vdb_upload(lancedb_uri=lancedb_uri, table_name=LANCEDB_TABLE, overwrite=True, create_index=True)
        )
    else:
        pdf_glob = str(input_dir / "*.pdf")
        ingestor = create_ingestor(run_mode="batch", ray_address=ray_address)
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
            )
            .vdb_upload(lancedb_uri=lancedb_uri, table_name=LANCEDB_TABLE, overwrite=True, create_index=True)
        )

    print("Running extraction...")
    ingest_start = time.perf_counter()
    ingestor.ingest(
        runtime_metrics_dir=str(runtime_metrics_dir) if runtime_metrics_dir is not None else None,
        runtime_metrics_prefix=runtime_metrics_prefix,
    )
    ingest_elapsed_s = time.perf_counter() - ingest_start
    processed_pages = _estimate_processed_pages(lancedb_uri, LANCEDB_TABLE)
    print("Extraction complete.")

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
            f"Recall stage requires LanceDB table {LANCEDB_TABLE!r} at {lancedb_uri!r}, "
            f"but it was not found."
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
            ext = ".txt" if input_type == "txt" else ".pdf"
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
            ext = ".txt" if input_type == "txt" else ".pdf"
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


if __name__ == "__main__":
    app()
