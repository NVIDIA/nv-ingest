import json
import logging
import os
import shutil
import time

from nv_ingest_client.client import Ingestor
from nv_ingest_client.util.document_analysis import analyze_document_chunks
from nv_ingest_client.util.milvus import nvingest_retrieval

from nv_ingest_harness.utils.interact import embed_info, kv_event_log, milvus_chunks, segment_results, pdf_page_count
from nv_ingest_harness.utils.vdb import get_lancedb_path

# Future: Will integrate with modular nv-ingest-harness-ingest when VDB upload is separated

# Suppress LazyLoadedList file not found errors to reduce terminal bloat
lazy_logger = logging.getLogger("nv_ingest_client.client.interface")
lazy_logger.setLevel(logging.CRITICAL)


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main test entry point.

    Args:
        config: TestConfig object with all settings
        log_path: Path for logging output

    Returns:
        Exit code (0 = success)
    """
    # Backward compatibility: if no config provided, we should error
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
        from nv_ingest_harness.utils.recall import get_recall_collection_name

        test_name = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
        collection_name = get_recall_collection_name(test_name)
    hostname = config.hostname
    sparse = config.sparse
    gpu_search = config.gpu_search

    # API version configuration
    api_version = config.api_version
    pdf_split_page_count = config.pdf_split_page_count

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
    enable_image_storage = config.enable_image_storage

    # Text splitting configuration
    split_chunk_size = config.split_chunk_size
    split_chunk_overlap = config.split_chunk_overlap

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
        caption_flags = []
        if config.caption_prompt:
            caption_flags.append("prompt override")
        if config.caption_reasoning is not None:
            caption_flags.append(f"reasoning={'on' if config.caption_reasoning else 'off'}")
        if caption_flags:
            pipeline_opts.append(f"caption ({', '.join(caption_flags)})")
        else:
            pipeline_opts.append("caption")
    if enable_split:
        pipeline_opts.append(f"text split: {split_chunk_size}/{split_chunk_overlap}")
    if enable_image_storage:
        pipeline_opts.append("image storage")

    if pipeline_opts:
        print(f"Pipeline: {', '.join(pipeline_opts)}")
    print("==========================")

    ingestion_start = time.time()

    # Build ingestor pipeline with API version configuration
    ingestor_kwargs = {"message_client_hostname": hostname, "message_client_port": 7670}
    if api_version == "v2":
        ingestor_kwargs["message_client_kwargs"] = {"api_version": "v2"}

    # Convert directory to recursive glob pattern to handle nested directories
    if os.path.isdir(data_dir):
        # Use **/*.pdf to recursively match PDF files in subdirectories
        file_pattern = os.path.join(data_dir, "**", "*.pdf")
    else:
        # Already a file or glob pattern
        file_pattern = data_dir

    ingestor = Ingestor(**ingestor_kwargs).files(file_pattern)

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
        caption_kwargs = {}
        if config.caption_prompt:
            caption_kwargs["prompt"] = config.caption_prompt
        if config.caption_reasoning is not None:
            caption_kwargs["reasoning"] = config.caption_reasoning
        ingestor = ingestor.caption(**caption_kwargs)

    if enable_split:
        ingestor = ingestor.split(
            chunk_size=split_chunk_size,
            chunk_overlap=split_chunk_overlap,
        )

    # Embed (must come before storage per pipeline ordering)
    ingestor = ingestor.embed(model_name=model_name)

    # Store images to disk (server-side image storage) - optional
    # Note: Supports both MinIO (s3://) and local disk (file://) via storage_uri
    # Config comes from test_configs.yaml or falls back to server defaults from environment
    if enable_image_storage:
        store_kwargs = {
            "structured": config.store_structured,
            "images": config.store_images,
        }
        # Pass optional storage config if specified
        if config.storage_uri:
            store_kwargs["storage_uri"] = config.storage_uri
        if config.storage_options:
            store_kwargs["storage_options"] = config.storage_options
        if config.public_base_url:
            store_kwargs["public_base_url"] = config.public_base_url

        ingestor = ingestor.store(**store_kwargs)

    # VDB upload and save results (respect vdb_backend)
    vdb_backend = config.vdb_backend
    lancedb_path = None
    if vdb_backend == "lancedb":
        lancedb_path = get_lancedb_path(config, collection_name)
        ingestor = ingestor.vdb_upload(
            vdb_op="lancedb",
            uri=lancedb_path,
            table_name=collection_name,
            purge_results_after_upload=False,
        )
    else:
        ingestor = ingestor.vdb_upload(
            vdb_op="milvus",
            collection_name=collection_name,
            dense_dim=dense_dim,
            sparse=sparse,
            gpu_search=gpu_search,
            model_name=model_name,
            purge_results_after_upload=False,
        )
    ingestor = ingestor.save_to_disk(output_directory=spill_dir)

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
    if vdb_backend != "lancedb":
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

    # Retrieval sanity (matching vdb_backend)
    queries = [
        "What is the dog doing and where?",
        "How many dollars does a power drill cost?",
    ]
    querying_start = time.time()
    if vdb_backend == "lancedb":
        try:
            from nv_ingest_client.util.vdb.lancedb import LanceDB
        except ImportError as exc:
            print(f"Warning: LanceDB retrieval not available ({exc}). Skipping retrieval sanity check.")
        else:
            lancedb_client = LanceDB(uri=lancedb_path, table_name=collection_name)
            _ = lancedb_client.retrieval(
                queries,
                embedding_endpoint=f"http://{hostname}:8012/v1",
                model_name=model_name,
                top_k=5,
            )
    else:
        _ = nvingest_retrieval(
            queries,
            collection_name,
            hybrid=sparse,
            embedding_endpoint=f"http://{hostname}:8012/v1",
            model_name=model_name,
            top_k=5,
            gpu_search=gpu_search,
            nv_ranker=False,
        )
    retrieval_time = time.time() - querying_start
    kv_event_log("retrieval_time_s", retrieval_time, log_path)

    # Summarize - Build comprehensive results dict
    dataset_name = os.path.basename(data_dir.rstrip("/")) if data_dir else "unknown"
    test_name = config.test_name or dataset_name

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
            "enable_image_storage": enable_image_storage,
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
