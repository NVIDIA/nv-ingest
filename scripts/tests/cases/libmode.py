"""
Library mode test script for nv-ingest.

This script demonstrates running the nv-ingest pipeline in library mode,
which is useful for local development, testing, and small-scale workloads.

Usage:
    python scripts/tests/cases/libmode.py

Requirements:
    - NVIDIA_API_KEY environment variable set
    - Python 3.12
    - nv-ingest packages installed
"""

import time

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline


def main():
    # Start the pipeline subprocess for library mode
    # quiet=True is the default when libmode=True, suppressing verbose startup logs
    pipeline = run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    # gpu_cagra accelerated indexing is not available in milvus-lite
    # Provide a filename for milvus_uri to use milvus-lite
    milvus_uri = "milvus.db"
    collection_name = "test"
    sparse = False

    # do content extraction from files
    ingestor = (
        Ingestor(client=client)
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=True,
            # Slower, but maximally accurate, especially for PDFs with pages that are scanned images
            # extract_method="nemoretriever_parse",
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            sparse=sparse,
            # for llama-3.2 embedder, use 1024 for e5-v5
            dense_dim=2048,
        )
    )

    print("Starting ingestion..")
    t0 = time.time()
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")

    # results blob is directly inspectable
    if results:
        print(ingest_json_results_to_blob(results[0]))

    if failures:
        print(f"There were {len(failures)} failures. Sample: {failures[0]}")

    # Clean up
    if pipeline:
        pipeline.stop()


# Required for multiprocessing on macOS (spawn start method)
if __name__ == "__main__":
    main()
