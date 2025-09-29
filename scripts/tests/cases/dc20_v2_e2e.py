import os
import sys
import time
import json
import logging
from datetime import datetime

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


def _now_timestr() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _get_env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _default_collection_name() -> str:
    # Make collection name configurable via TEST_NAME env var, default to dc20_v2
    test_name = _get_env("TEST_NAME", "dc20_v2")
    return f"{test_name}_{_now_timestr()}"


def main() -> int:
    # Set API version to V2 (can be overridden by environment)
    if "NV_INGEST_API_VERSION" not in os.environ:
        os.environ["NV_INGEST_API_VERSION"] = "v2"

    # Dataset-agnostic: no hardcoded paths, configurable via environment
    data_dir = _get_env("DATASET_DIR")
    if not data_dir:
        print("ERROR: DATASET_DIR environment variable is required")
        print("Example: DATASET_DIR=/datasets/bo20 python dc20_v2_e2e.py")
        return 2

    if not os.path.isdir(data_dir):
        print(f"ERROR: Dataset directory does not exist: {data_dir}")
        print("Please check the DATASET_DIR path and ensure it's accessible")
        return 2

    spill_dir = _get_env("SPILL_DIR", "/tmp/spill")
    os.makedirs(spill_dir, exist_ok=True)

    collection_name = _get_env("COLLECTION_NAME", _default_collection_name())
    hostname = _get_env("HOSTNAME", "localhost")
    sparse = _get_env("SPARSE", "true").lower() == "true"
    gpu_search = _get_env("GPU_SEARCH", "false").lower() == "true"

    # Extraction configuration from environment variables
    extract_text = _get_env("EXTRACT_TEXT", "true").lower() == "true"
    extract_tables = _get_env("EXTRACT_TABLES", "true").lower() == "true"
    extract_charts = _get_env("EXTRACT_CHARTS", "true").lower() == "true"
    extract_images = _get_env("EXTRACT_IMAGES", "false").lower() == "true"
    text_depth = _get_env("TEXT_DEPTH", "page")
    table_output_format = _get_env("TABLE_OUTPUT_FORMAT", "markdown")
    extract_infographics = _get_env("EXTRACT_INFOGRAPHICS", "true").lower() == "true"

    # Logging configuration
    log_path = _get_env("LOG_PATH", "test_results")

    model_name, dense_dim = embed_info()

    # Get actual API version being used
    actual_api_version = os.environ.get("NV_INGEST_API_VERSION", "v1")

    # Log configuration for transparency
    print("=== Configuration ===")
    print(f"Dataset: {data_dir}")
    print(f"Collection: {collection_name}")
    print(f"Spill: {spill_dir}")
    print(f"Hostname: {hostname}")
    print(f"API Version: {actual_api_version} (expected: v2)")
    print(f"Embed model: {model_name}, dim: {dense_dim}")
    print(f"Sparse: {sparse}, GPU search: {gpu_search}")
    print(f"Extract text: {extract_text}, tables: {extract_tables}, charts: {extract_charts}")
    print(f"Extract images: {extract_images}, infographics: {extract_infographics}")
    print(f"Text depth: {text_depth}, table format: {table_output_format}")
    print("==============================")

    ingestion_start = time.time()

    # Create Ingestor with V2 API endpoints
    ingestor = (
        Ingestor(
            message_client_hostname=hostname,
            message_client_port=7670,
            # The API version is set via environment variable above
        )
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
        .embed(model_name=model_name)
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
    if _get_env("DOC_ANALYSIS", "false").lower() == "true":
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
    test_name = _get_env("TEST_NAME", "dc20_v2")
    summary = {
        "test_name": test_name,
        "api_version": "v2",
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
        "text_chunks": sum(len(x) for x in text_results),
        "table_chunks": sum(len(x) for x in table_results),
        "chart_chunks": sum(len(x) for x in chart_results),
        "dataset_pages": total_pages,
        "pages_per_second": pages_per_second if total_pages > 0 and ingestion_time > 0 else None,
    }
    print(f"\n{test_name}_e2e summary:")
    print(json.dumps(summary, indent=2))

    # Compare with expected V1 results if available
    print("\n=== V2 API Results ===")
    print(f"Text chunks: {sum(len(x) for x in text_results)}")
    print(f"Table chunks: {sum(len(x) for x in table_results)}")
    print(f"Chart chunks: {sum(len(x) for x in chart_results)}")

    # For dc20/bo20 dataset, expected values from V1:
    # text: 496, tables: 164, charts: 184
    if "bo20" in data_dir or "dc20" in data_dir:
        print("\nExpected from V1 (bo20/dc20): text=496, tables=164, charts=184")
        text_count = sum(len(x) for x in text_results)
        table_count = sum(len(x) for x in table_results)
        chart_count = sum(len(x) for x in chart_results)
        if text_count == 496 and table_count == 164 and chart_count == 184:
            print("✓ V2 results match V1 expected values!")
        else:
            print("✗ V2 results differ from V1 expected values")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
