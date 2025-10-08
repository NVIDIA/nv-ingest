import os
import json
import logging
import sys
import time
from pathlib import Path

from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import nvingest_retrieval
from nv_ingest_client.util.document_analysis import analyze_document_chunks

# Import from interact module (now properly structured)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from interact import embed_info, milvus_chunks, segment_results, kv_event_log, pdf_page_count  # noqa: E402

# Future: Will integrate with modular ingest_documents.py when VDB upload is separated

# Suppress LazyLoadedList file not found errors to reduce terminal bloat
lazy_logger = logging.getLogger("nv_ingest_client.client.interface")
lazy_logger.setLevel(logging.CRITICAL)

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # Optional; stats logging will be skipped if unavailable

from utils import default_collection_name, get_repo_root


def main() -> int:
    # Dataset-agnostic: no hardcoded paths, configurable via environment
    data_dir = os.getenv("DATASET_DIR")
    if not data_dir:
        print("ERROR: DATASET_DIR environment variable is required")
        print("Example: DATASET_DIR=/datasets/bo20 python e2e.py")
        return 2

    if not os.path.isdir(data_dir):
        print(f"ERROR: Dataset directory does not exist: {data_dir}")
        print("Please check the DATASET_DIR path and ensure it's accessible")
        return 2

    spill_dir = os.getenv("SPILL_DIR", "/tmp/spill")
    os.makedirs(spill_dir, exist_ok=True)

    collection_name = os.getenv("COLLECTION_NAME", default_collection_name())
    hostname = os.getenv("HOSTNAME", "localhost")
    sparse = os.getenv("SPARSE", "true").lower() == "true"
    gpu_search = os.getenv("GPU_SEARCH", "false").lower() == "true"

    # Extraction configuration (core testing variables)
    extract_text = os.getenv("EXTRACT_TEXT", "true").lower() == "true"
    extract_tables = os.getenv("EXTRACT_TABLES", "true").lower() == "true"
    extract_charts = os.getenv("EXTRACT_CHARTS", "true").lower() == "true"
    extract_images = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"
    extract_infographics = os.getenv("EXTRACT_INFOGRAPHICS", "true").lower() == "true"
    text_depth = os.getenv("TEXT_DEPTH", "page")
    table_output_format = os.getenv("TABLE_OUTPUT_FORMAT", "markdown")

    # Optional pipeline steps (for special testing scenarios)
    enable_caption = os.getenv("ENABLE_CAPTION", "false").lower() == "true"
    enable_split = os.getenv("ENABLE_SPLIT", "false").lower() == "true"

    # Logging configuration
    log_path = os.getenv("LOG_PATH", "test_results")

    # UDF and LLM Summaries
    udf_path = Path(get_repo_root()) / "api/src/udfs/llm_summarizer_udf.py:content_summarizer"
    print(f"Path to User-Defined Function: {str(udf_path)}")
    llm_model = os.getenv("LLM_SUMMARIZATION_MODEL", "nvdev/nvidia/llama-3.1-nemotron-70b-instruct")

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
    if os.getenv("DOC_ANALYSIS", "false").lower() == "true":
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
    test_name = os.getenv("TEST_NAME", dataset_name)
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
    os.rmdir(spill_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
