import json
import logging
import os
import shutil
import sys
import time

from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.document_analysis import analyze_document_chunks
from nv_ingest_client.util.milvus import nvingest_retrieval

# Import from interact module (now properly structured)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from interact import embed_info, kv_event_log, milvus_chunks, segment_results, pdf_page_count

# Future: Will integrate with modular ingest_documents.py when VDB upload is separated

# Suppress LazyLoadedList file not found errors to reduce terminal bloat
lazy_logger = logging.getLogger("nv_ingest_client.client.interface")
lazy_logger.setLevel(logging.CRITICAL)

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None  # Optional; stats logging will be skipped if unavailable

from utils import default_collection_name


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

    collection_name = os.getenv("COLLECTION_NAME") or default_collection_name()
    hostname = os.getenv("HOSTNAME", "localhost")
    sparse = os.getenv("SPARSE", "true").lower() == "true"
    gpu_search = os.getenv("GPU_SEARCH", "false").lower() == "true"

    # API version configuration (v1 = default, v2 = PDF splitting support)
    api_version = os.getenv("API_VERSION", "v1").lower()

    # PDF split configuration (V2 only - server-side page splitting)
    pdf_split_page_count = os.getenv("PDF_SPLIT_PAGE_COUNT")
    if pdf_split_page_count:
        try:
            pdf_split_page_count = int(pdf_split_page_count)
            if api_version != "v2":
                print(f"WARNING: PDF_SPLIT_PAGE_COUNT={pdf_split_page_count} is set but API_VERSION={api_version}")
                print("         PDF splitting only works with API_VERSION=v2. This setting will be ignored.")
                pdf_split_page_count = None
        except ValueError:
            print(f"WARNING: Invalid PDF_SPLIT_PAGE_COUNT value '{pdf_split_page_count}', ignoring")
            pdf_split_page_count = None

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

    # Text splitting configuration (client-side text chunking)
    split_chunk_size = int(os.getenv("SPLIT_CHUNK_SIZE", "1024"))
    split_chunk_overlap = int(os.getenv("SPLIT_CHUNK_OVERLAP", "150"))

    # Logging configuration
    log_path = os.getenv("LOG_PATH", "test_results")

    model_name, dense_dim = embed_info()

    # Log configuration for transparency
    print("=== Test Configuration ===")
    print(f"Dataset: {data_dir}")
    print(f"Collection: {collection_name}")
    print(f"Embed: {model_name} (dim={dense_dim}, sparse={sparse})")

    # Extraction config
    extractions = []
    if extract_text:
        extractions.append("text")
    if extract_tables:
        extractions.append("tables")
    if extract_charts:
        extractions.append("charts")
    if extract_images:
        extractions.append("images")
    if extract_infographics:
        extractions.append("infographics")
    print(f"Extract: {', '.join(extractions)}")

    # Pipeline options
    pipeline_opts = []
    if api_version == "v2" and pdf_split_page_count:
        clamped_value = max(1, min(pdf_split_page_count, 128))
        if clamped_value != pdf_split_page_count:
            pipeline_opts.append(f"PDF split: {pdf_split_page_count} pages (clamped to {clamped_value})")
        else:
            pipeline_opts.append(f"PDF split: {pdf_split_page_count} pages")
    elif api_version == "v2":
        pipeline_opts.append("PDF split: 32 pages (default)")

    if enable_caption:
        pipeline_opts.append("caption")
    if enable_split:
        pipeline_opts.append(f"text split: {split_chunk_size}/{split_chunk_overlap}")

    if pipeline_opts:
        print(f"Pipeline: {', '.join(pipeline_opts)}")
    print("==========================")

    ingestion_start = time.time()

    # Build ingestor pipeline with API version configuration
    ingestor_kwargs = {"message_client_hostname": hostname, "message_client_port": 7670}
    if api_version == "v2":
        ingestor_kwargs["message_client_kwargs"] = {"api_version": "v2"}

    ingestor = Ingestor(**ingestor_kwargs).files(data_dir)

    # V2-only: Configure PDF splitting (server-side page splitting)
    if api_version == "v2" and pdf_split_page_count:
        ingestor = ingestor.pdf_split_config(pages_per_chunk=pdf_split_page_count)

    # Extraction step
    ingestor = ingestor.extract(
        extract_text=extract_text,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_images=extract_images,
        text_depth=text_depth,
        table_output_format=table_output_format,
        extract_infographics=extract_infographics,
    )

    # Optional pipeline steps
    if enable_caption:
        ingestor = ingestor.caption()

    if enable_split:
        ingestor = ingestor.split(
            chunk_size=split_chunk_size,
            chunk_overlap=split_chunk_overlap,
        )

    # Embed and upload (core pipeline)
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
    retrieval_time = time.time() - querying_start
    kv_event_log("retrieval_time_s", retrieval_time, log_path)

    # Summarize - Build comprehensive results dict
    dataset_name = os.path.basename(data_dir.rstrip("/")) if data_dir else "unknown"
    test_name = os.getenv("TEST_NAME", dataset_name)

    # Structure results for consolidation with runner metadata
    test_results = {
        "test_config": {
            "test_name": test_name,
            "api_version": api_version,
            "dataset_dir": data_dir,
            "collection_name": collection_name,
            "hostname": hostname,
            "model_name": model_name,
            "dense_dim": dense_dim,
            "sparse": sparse,
            "gpu_search": gpu_search,
            "extract_text": extract_text,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_images": extract_images,
            "extract_infographics": extract_infographics,
            "text_depth": text_depth,
            "table_output_format": table_output_format,
            "enable_caption": enable_caption,
            "enable_split": enable_split,
        },
        "results": {
            "result_count": len(results),
            "failure_count": len(failures),
            "ingestion_time_s": ingestion_time,
            "total_pages": total_pages,
            "pages_per_second": pages_per_second,
            "text_chunks": sum(len(x) for x in text_results),
            "table_chunks": sum(len(x) for x in table_results),
            "chart_chunks": sum(len(x) for x in chart_results),
            "retrieval_time_s": retrieval_time,
        },
    }

    # Add split config if enabled
    if enable_split:
        test_results["test_config"]["split_chunk_size"] = split_chunk_size
        test_results["test_config"]["split_chunk_overlap"] = split_chunk_overlap

    print(f"\n{test_name}_e2e summary:")
    print(json.dumps(test_results, indent=2))

    # Write test results for run.py to consolidate
    results_file = os.path.join(log_path, "_test_results.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nRemoving spill directory: {spill_dir}")
    shutil.rmtree(spill_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
