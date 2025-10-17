import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Optional

import requests

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


def _default_collection_name() -> str:
    # Make collection name configurable via TEST_NAME env var, default to dc20_v2
    test_name = os.getenv("TEST_NAME") or "dc20_v2"
    return f"{test_name}_{_now_timestr()}"


def _expected_counts_for_dataset(dataset_path: str) -> dict[str, int] | None:
    dataset_path_lower = dataset_path.lower()
    if any(marker in dataset_path_lower for marker in ("bo20", "dc20")):
        return {"text": 496, "tables": 164, "charts": 184}
    return None


def _assert_counts_match(expected: dict[str, int], actual: dict[str, int]) -> None:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

    if mismatches:
        mismatch_details = "; ".join(mismatches)
        raise AssertionError("Chunk count assertions failed compared to V1 baseline: " + mismatch_details)


def main() -> int:
    # Dataset-agnostic: no hardcoded paths, configurable via environment
    data_dir = os.getenv("DATASET_DIR")
    if not data_dir:
        print("ERROR: DATASET_DIR environment variable is required")
        print("Example: DATASET_DIR=/datasets/bo20 python dc20_v2_e2e.py")
        return 2

    if not os.path.isdir(data_dir):
        print(f"ERROR: Dataset directory does not exist: {data_dir}")
        print("Please check the DATASET_DIR path and ensure it's accessible")
        return 2

    spill_dir = os.getenv("SPILL_DIR") or "/tmp/spill"
    os.makedirs(spill_dir, exist_ok=True)

    collection_name = os.getenv("COLLECTION_NAME") or _default_collection_name()
    hostname = os.getenv("HOSTNAME") or "localhost"
    sparse = (os.getenv("SPARSE") or "true").lower() == "true"
    gpu_search = (os.getenv("GPU_SEARCH") or "false").lower() == "true"

    # Extraction configuration from environment variables
    extract_text = (os.getenv("EXTRACT_TEXT") or "true").lower() == "true"
    extract_tables = (os.getenv("EXTRACT_TABLES") or "true").lower() == "true"
    extract_charts = (os.getenv("EXTRACT_CHARTS") or "true").lower() == "true"
    extract_images = (os.getenv("EXTRACT_IMAGES") or "false").lower() == "true"
    text_depth = os.getenv("TEXT_DEPTH") or "page"
    table_output_format = os.getenv("TABLE_OUTPUT_FORMAT") or "markdown"
    extract_infographics = (os.getenv("EXTRACT_INFOGRAPHICS") or "true").lower() == "true"

    # PDF splitting configuration (only override if explicitly set)
    pdf_split_page_count = os.getenv("PDF_SPLIT_PAGE_COUNT")
    if pdf_split_page_count:
        try:
            pdf_split_page_count = int(pdf_split_page_count)
        except ValueError:
            print(f"Warning: Invalid PDF_SPLIT_PAGE_COUNT value, ignoring: {pdf_split_page_count}")
            pdf_split_page_count = None

    # Logging configuration
    log_path = os.getenv("LOG_PATH") or "test_results"

    model_name, dense_dim = embed_info()

    # This test always uses V2 API
    actual_api_version = "v2"
    assert_v1_baseline = (os.getenv("ASSERT_V1_BASELINE") or "false").lower() == "true"

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

    ## Displaying service side logic for PDF splitting for V2
    if pdf_split_page_count:
        # Show clamping info if value is out of bounds
        clamped_value = max(1, min(pdf_split_page_count, 128))
        if clamped_value != pdf_split_page_count:
            print(f"PDF split page count: {pdf_split_page_count} (will be clamped to {clamped_value} by server)")
        else:
            print(f"PDF split page count: {pdf_split_page_count}")
    else:
        print("PDF split page count: Using server default (32)")

    print(f"Assert V1 baseline counts: {assert_v1_baseline}")
    print("==============================")

    ingestion_start = time.time()

    # Create Ingestor with V2 API endpoints (explicit configuration)
    ingestor = Ingestor(
        message_client_hostname=hostname,
        message_client_port=7670,
        message_client_kwargs={"api_version": "v2"},
    ).files(data_dir)

    # Optional: Configure PDF splitting (comment out to use server default)
    # Set via environment variable OR uncomment line below for quick testing
    if pdf_split_page_count:
        ingestor = ingestor.pdf_split_config(pages_per_chunk=pdf_split_page_count)
    # ingestor = ingestor.pdf_split_config(pages_per_chunk=2)  # Uncomment to override

    ingestor = (
        ingestor.extract(
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
            purge_results_after_upload=False,  # Leave chunks intact so the baseline assertions can run
        )
        .save_to_disk(output_directory=spill_dir)
    )

    results, failures, parent_trace_ids = ingestor.ingest(
        show_progress=True,
        return_failures=True,
        save_to_disk=True,
        include_parent_trace_ids=True,
    )
    ingestion_time = time.time() - ingestion_start
    if not results:
        raise AssertionError("Ingestion returned zero results; expected at least one chunk.")

    kv_event_log("result_count", len(results), log_path)
    kv_event_log("failure_count", len(failures), log_path)
    kv_event_log("ingestion_time_s", ingestion_time, log_path)

    total_pages = pdf_page_count(data_dir)
    pages_per_second = None
    if total_pages > 0 and ingestion_time > 0:
        pages_per_second = total_pages / ingestion_time
        kv_event_log("pages_per_second", pages_per_second, log_path)

    enable_trace_debug = (os.getenv("TRACE_DEBUG") or "false").lower() == "true"

    # Fetch the parent job response to inspect aggregated telemetry (trace debug only)
    parent_job_ids: list[str] = list(parent_trace_ids)
    if enable_trace_debug and parent_job_ids:
        print(f"\nDiscovered parent job candidates: {parent_job_ids}")

        artifacts_dir_env = os.getenv("TRACE_ARTIFACT_DIR")
        artifacts_dir: Optional[str] = None
        if artifacts_dir_env:
            artifacts_dir = os.path.join(artifacts_dir_env, "parents") if len(parent_job_ids) > 1 else artifacts_dir_env
            os.makedirs(artifacts_dir, exist_ok=True)

        for parent_job_id in parent_job_ids:
            try:
                parent_response = requests.get(
                    f"http://{hostname}:7670/v2/fetch_job/{parent_job_id}",
                    timeout=30,
                )
                print(f"\nParent job fetch status: {parent_response.status_code} (job_id={parent_job_id})")

                if parent_response.ok:
                    parent_payload = parent_response.json()
                    parent_metadata = parent_payload.get("metadata", {})

                    trace_segments = parent_metadata.get("trace_segments")
                    annotation_segments = parent_metadata.get("annotation_segments")

                    trace_count = len(trace_segments) if isinstance(trace_segments, list) else "n/a"
                    annotation_count = len(annotation_segments) if isinstance(annotation_segments, list) else "n/a"
                    print(f"Parent trace segments: {trace_count}")
                    print(f"Parent annotation segments: {annotation_count}")

                    if isinstance(trace_segments, list) and trace_segments:
                        print(f"Sample parent trace segment: {trace_segments[0]}")
                    if isinstance(annotation_segments, list) and annotation_segments:
                        print(f"Sample parent annotation segment: {annotation_segments[0]}")

                    if artifacts_dir:
                        output_path = os.path.join(artifacts_dir, f"{parent_job_id}_fetch.json")
                        with open(output_path, "w", encoding="utf-8") as fp:
                            json.dump(parent_payload, fp, indent=2)
                        print(f"Persisted parent fetch payload to {output_path}")

                else:
                    print(f"Parent job fetch failed: {parent_response.text}")
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"Failed to retrieve parent job telemetry for {parent_job_id}: {exc}")

    if failures:
        raise AssertionError(f"Ingestion produced {len(failures)} failures: {failures}")

    # Optional: log chunk stats and per-type breakdown
    milvus_chunks(f"http://{hostname}:19530", collection_name)
    text_results, table_results, chart_results = segment_results(results)
    text_chunk_count = sum(len(x) for x in text_results)
    table_chunk_count = sum(len(x) for x in table_results)
    chart_chunk_count = sum(len(x) for x in chart_results)

    kv_event_log("text_chunks", text_chunk_count, log_path)
    kv_event_log("table_chunks", table_chunk_count, log_path)
    kv_event_log("chart_chunks", chart_chunk_count, log_path)

    # Document-level analysis
    if (os.getenv("DOC_ANALYSIS") or "false").lower() == "true":
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
    test_name = os.getenv("TEST_NAME") or "dc20_v2"
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
        "text_chunks": text_chunk_count,
        "table_chunks": table_chunk_count,
        "chart_chunks": chart_chunk_count,
        "dataset_pages": total_pages,
        "pages_per_second": pages_per_second if total_pages > 0 and ingestion_time > 0 else None,
        "assert_v1_baseline": assert_v1_baseline,
    }
    print(f"\n{test_name}_e2e summary:")
    print(json.dumps(summary, indent=2))

    # Compare with expected V1 results if available
    if assert_v1_baseline:
        expected_counts = _expected_counts_for_dataset(data_dir)
        if not expected_counts:
            raise AssertionError(
                "ASSERT_V1_BASELINE is true but no baseline expectations are defined for this dataset."
            )

        actual_counts = {
            "text": text_chunk_count,
            "tables": table_chunk_count,
            "charts": chart_chunk_count,
        }
        _assert_counts_match(expected_counts, actual_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
