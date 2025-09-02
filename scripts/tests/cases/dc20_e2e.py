import os
import sys
import time
import json
import logging
from datetime import datetime

from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.milvus import nvingest_retrieval

# Import from interact module (now properly structured)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from interact import embed_info, milvus_chunks, segment_results, kv_event_log  # noqa: E402

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
    # Make collection name configurable via TEST_NAME env var, default to dc20
    test_name = _get_env("TEST_NAME", "dc20")
    return f"{test_name}_{_now_timestr()}"


def main() -> int:
    # Dataset-agnostic: no hardcoded paths, configurable via environment
    data_dir = _get_env("DATASET_DIR")
    if not data_dir:
        print("ERROR: DATASET_DIR environment variable is required")
        print("Example: DATASET_DIR=/datasets/bo20 python dc20_e2e.py")
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

    # Log configuration for transparency
    print("=== Configuration ===")
    print(f"Dataset: {data_dir}")
    print(f"Collection: {collection_name}")
    print(f"Spill: {spill_dir}")
    print(f"Hostname: {hostname}")
    print(f"Embed model: {model_name}, dim: {dense_dim}")
    print(f"Sparse: {sparse}, GPU search: {gpu_search}")
    print(f"Extract text: {extract_text}, tables: {extract_tables}, charts: {extract_charts}")
    print(f"Extract images: {extract_images}, infographics: {extract_infographics}")
    print(f"Text depth: {text_depth}, table format: {table_output_format}")
    print("====================")

    ingestion_start = time.time()

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
        .embed(model_name=model_name)
        .vdb_upload(
            collection_name=collection_name,
            dense_dim=dense_dim,
            sparse=sparse,
            gpu_search=gpu_search,
            model_name=model_name,
        )
        .save_to_disk(output_directory=spill_dir)
    )

    results, failures = ingestor.ingest(show_progress=True, return_failures=True, save_to_disk=True)
    ingestion_time = time.time() - ingestion_start
    kv_event_log("result_count", len(results), log_path)
    kv_event_log("failure_count", len(failures), log_path)
    kv_event_log("ingestion_time_s", ingestion_time, log_path)

    # Optional: log chunk stats and per-type breakdown
    milvus_chunks(f"http://{hostname}:19530", collection_name)
    text_results, table_results, chart_results = segment_results(results)
    kv_event_log("text_chunks", sum(len(x) for x in text_results), log_path)
    kv_event_log("table_chunks", sum(len(x) for x in table_results), log_path)
    kv_event_log("chart_chunks", sum(len(x) for x in chart_results), log_path)

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
    test_name = _get_env("TEST_NAME", "dc20")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
