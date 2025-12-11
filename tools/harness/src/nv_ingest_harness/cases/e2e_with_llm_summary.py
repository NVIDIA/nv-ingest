import os
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import nvingest_retrieval
from nv_ingest_client.util.document_analysis import analyze_document_chunks

from nv_ingest_harness.interact import embed_info, milvus_chunks, segment_results, kv_event_log, pdf_page_count  # noqa: E402

# Future: Will integrate with modular ingest_documents.py when VDB upload is separated

# Suppress LazyLoadedList file not found errors to reduce terminal bloat
lazy_logger = logging.getLogger("nv_ingest_client.client.interface")
lazy_logger.setLevel(logging.CRITICAL)

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # Optional; stats logging will be skipped if unavailable

from utils import get_repo_root


def main(config=None, log_path: str = "test_results") -> int:
    """
    E2E test with LLM summarization via UDF.

    Args:
        config: TestConfig object with all settings
        log_path: Path for logging output

    Returns:
        Exit code (0 = success)
    """
    # Backward compatibility: if no config provided, error
    if config is None:
        print("ERROR: No configuration provided")
        print("This test case requires a config object from the test runner")
        return 2

    # Extract configuration from config object
    data_dir = config.dataset_dir
    spill_dir = config.spill_dir
    os.makedirs(spill_dir, exist_ok=True)

    # Use consistent collection naming with recall pattern
    # If collection_name not set, generate from test_name or dataset basename
    if config.collection_name:
        collection_name = config.collection_name
    else:
        from recall_utils import get_recall_collection_name

        test_name = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
        collection_name = get_recall_collection_name(test_name)
    hostname = config.hostname
    sparse = config.sparse
    gpu_search = config.gpu_search

    # Extraction configuration
    extract_text = config.extract_text
    extract_tables = config.extract_tables
    extract_charts = config.extract_charts
    extract_images = config.extract_images
    extract_infographics = config.extract_infographics
    text_depth = config.text_depth
    table_output_format = config.table_output_format

    # Optional pipeline steps
    enable_caption = config.enable_caption
    enable_split = config.enable_split

    # UDF and LLM Summaries
    udf_path = Path(get_repo_root()) / "api/src/udfs/llm_summarizer_udf.py:content_summarizer"
    print(f"Path to User-Defined Function: {str(udf_path)}")
    llm_model = config.llm_summarization_model

    model_name, dense_dim = embed_info()

    # Log configuration for transparency
    print("=== Configuration ===")
    print(f"Dataset: {data_dir}")
    print(f"Collection: {collection_name}")
    print(f"Hostname: {hostname}")
    print(f"Embed model: {model_name}, dim: {dense_dim}")
    print(f"LLM Summarize Model: {llm_model}")
    print(f"Sparse: {sparse}, GPU search: {gpu_search}")
    print(f"Extract text: {extract_text}, tables: {extract_tables}, charts: {extract_charts}")
    print(f"Extract images: {extract_images}, infographics: {extract_infographics}")
    print(f"Text depth: {text_depth}, table format: {table_output_format}")
    if enable_caption:
        print("Caption: enabled")
    if enable_split:
        print("Split: enabled")
    print("====================")

    ingestion_start = time.time()

    # Build ingestor pipeline (simplified)
    ingestor = (
        Ingestor(message_client_hostname=hostname, message_client_port=7670)
        .files(data_dir)
        .extract(
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_images=extract_images,
            text_depth=text_depth,
            table_output_format=table_output_format,
            extract_infographics=extract_infographics,
        )
        .udf(udf_function=str(udf_path), target_stage="text_splitter", run_after=True)
    )

    # Optional pipeline steps
    if enable_caption:
        ingestor = ingestor.caption()

    if enable_split:
        ingestor = ingestor.split()

    # Embed and upload (core pipeline)
    print("Uploading to collection:", collection_name)
    ingestor = (
        ingestor.embed(model_name=model_name)
        .vdb_upload(
            collection_name=collection_name,
            dense_dim=dense_dim,
            sparse=sparse,
            gpu_search=gpu_search,
            model_name=model_name,
            purge_results_after_upload=False,
        )
        .save_to_disk(output_directory=spill_dir)
    )

    results, failures = ingestor.ingest(show_progress=True, return_failures=True, save_to_disk=True)
    ingestion_time = time.time() - ingestion_start
    kv_event_log("result_count", len(results), log_path)
    kv_event_log("failure_count", len(failures), log_path)
    kv_event_log("ingestion_time_s", ingestion_time, log_path)

    total_pages = pdf_page_count(data_dir)
    pages_per_second = None
    if total_pages > 0 and ingestion_time > 0:
        pages_per_second = total_pages / ingestion_time
        kv_event_log("pages_per_second", pages_per_second, log_path)

    # Optional: log chunk stats and per-type breakdown
    milvus_chunks(f"http://{hostname}:19530", collection_name)
    text_results, table_results, chart_results = segment_results(results)
    kv_event_log("text_chunks", sum(len(x) for x in text_results), log_path)
    kv_event_log("table_chunks", sum(len(x) for x in table_results), log_path)
    kv_event_log("chart_chunks", sum(len(x) for x in chart_results), log_path)

    # Document-level analysis
    if os.getenv("DOC_ANALYSIS", "false").lower() == "true":  # DOC_ANALYSIS set by run.py
        print("\nDocument Analysis:")
        document_breakdown = analyze_document_chunks(results)

        if document_breakdown:
            # Show individual documents
            for doc_name, pages in document_breakdown.items():
                total_counts = pages["total"]
                total_elements = sum(total_counts.values())
                print(
                    f"  {doc_name}: {total_elements} elements "
                    f"(text: {total_counts['text']}, tables: {total_counts['tables']}, "
                    f"charts: {total_counts['charts']}, images: {total_counts['unstructured_images']}, "
                    f"infographics: {total_counts['infographics']})"
                )
        else:
            print("  No document data available")

    # Retrieval sanity
    queries = [
        "What is the dog doing and where?",
        "How many dollars does a power drill cost?",
    ]
    querying_start = time.time()
    _ = nvingest_retrieval(
        queries,
        collection_name,
        hybrid=sparse,
        embedding_endpoint=f"http://{hostname}:8012/v1",
        embedding_model_name=model_name,
        model_name=model_name,
        top_k=5,
        gpu_search=gpu_search,
    )
    kv_event_log("retrieval_time_s", time.time() - querying_start, log_path)

    # Summarize
    dataset_name = os.path.basename(data_dir.rstrip("/")) if data_dir else "unknown"
    test_name = config.test_name or dataset_name
    summary = {
        "test_name": test_name,
        "dataset_dir": data_dir,
        "collection_name": collection_name,
        "hostname": hostname,
        "model_name": model_name,
        "dense_dim": dense_dim,
        "sparse": sparse,
        "gpu_search": gpu_search,
        "ingestion_time_s": ingestion_time,
        "result_count": len(results),
        "failure_count": len(failures),
    }
    print(f"{test_name}_e2e summary:")
    print(json.dumps(summary, indent=2))

    print(f"Removing spill directory: {spill_dir}")
    shutil.rmtree(spill_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
