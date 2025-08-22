#!/usr/bin/env python3
"""
Comprehensive benchmark script for testing nv-ingest with and without store task.

Measures impact of storing images/structured artifacts to MinIO on throughput and latency.
Provides detailed performance metrics and image analysis for store task evaluation.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime

# Add utils path - adjusted for scripts/tests/cases location
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "interact"))

# Third-party and local imports (after path modification)
from nv_ingest_client.client import Ingestor  # noqa: E402
from nv_ingest_client.util.milvus import nvingest_retrieval  # noqa: E402
from nv_ingest_client.util.image_disk_utils import save_images_from_ingestor_results  # noqa: E402
from utils import (  # noqa: E402
    pdf_page_count,
    milvus_chunks,
    segment_results,
    embed_info,
    date_hour,
    get_parent_script_filename,
)

# Add log function
logs = []


def log(key, val):
    print(f"{date_hour()}: {key}: {val}")
    caller_fn = get_parent_script_filename()

    log_fn = "test_results/" + caller_fn.split(".")[0] + ".json"
    logs.append({key: val})
    with open(log_fn, "w") as fp:
        fp.write(json.dumps(logs))


class StoreBenchmark:
    """
    Comprehensive benchmark class for testing nv-ingest store task performance.

    This class compares ingestion performance with and without the store task enabled,
    measuring the impact on throughput, latency, and resource utilization.

    Attributes:
        dataset_name (str): Name of the dataset being benchmarked
        dataset_path (str): Path to the dataset files
        hostname (str): Hostname for nv-ingest services
        results (dict): Collected benchmark results
        run_id (str): Unique identifier for this benchmark run
    """

    def __init__(self, dataset_name, dataset_path, hostname="localhost"):
        """
        Initialize benchmark with dataset and configuration.

        Args:
            dataset_name (str): Name identifier for the dataset
            dataset_path (str): Path to directory containing PDF files
            hostname (str): Hostname for nv-ingest services
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.hostname = hostname
        self.results = {}
        self.run_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get embedding model info
        self.model_name, self.dense_dim = embed_info()
        print(f"Using embedding model: {self.model_name}, dim: {self.dense_dim}")

        # Count files and pages for benchmarking context
        self.file_count = len(os.listdir(dataset_path))
        self.page_count = pdf_page_count(dataset_path)
        print(f"Dataset: {self.file_count} files, {self.page_count} pages")

    def create_ingestor_base(self, collection_name, store_enabled=False, store_params=None):
        """
        Create base ingestor configuration with consistent settings.

        Args:
            collection_name (str): Name for the Milvus collection
            store_enabled (bool): Whether to enable MinIO store task
            store_params (dict): Configuration parameters for store task

        Returns:
            Ingestor: Configured ingestor instance ready for ingestion
        """
        # Set up temporary file directory in /raid instead of /tmp
        temp_dir = f"/raid/jioffe/tmp/store_task/{self.run_id}_{collection_name}"

        ingestor = (
            Ingestor(message_client_hostname=self.hostname, message_client_port=7670)
            .files(self.dataset_path)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                # extract_images is for unstructured images only;
                # store task works with structured images
                extract_images=False,
                extract_infographics=False,
                # Can be toggled for ablation studies
                extract_page_as_image=False,
                text_depth="page",
                table_output_format="markdown",
            )
            .embed(
                model_name=self.model_name,
            )
            .save_to_disk(output_directory=temp_dir, cleanup=False)
        )

        # Add processing tasks in the correct order
        ingestor = ingestor.dedup().filter().split()

        # Add store task if enabled
        if store_enabled:
            if not store_params:
                raise ValueError("store_params must be provided when store_enabled=True")

            # Ensure params dict exists and has required MinIO credentials
            params = store_params.get("params", {})
            if not params.get("access_key") or not params.get("secret_key"):
                raise ValueError("store_params.params must include 'access_key' and 'secret_key'")

            # Add collection name to store params so store task knows which collection to use
            params["collection_name"] = collection_name

            ingestor = ingestor.store(
                structured=store_params.get("structured", True), images=store_params.get("images", True), params=params
            )

        ingestor = ingestor.vdb_upload(
            collection_name=collection_name,
            dense_dim=self.dense_dim,
            sparse=True,  # ablation possibility?
            gpu_search=True,  # ablation possibility?
            model_name=self.model_name,
            purge_results_after_upload=False,
            stream=True,
            # Force bulk vs streaming - if < threshold then streaming
            threshold=1000,
        )

        return ingestor

    def run_benchmark(self, condition_name, store_enabled=False, store_params=None, skip_image_analysis=False):
        """
        Execute a single benchmark condition and collect comprehensive metrics.

        Args:
            condition_name (str): Descriptive name for this benchmark condition
            store_enabled (bool): Whether to enable MinIO store task
            store_params (dict): MinIO configuration parameters
            skip_image_analysis (bool): Skip saving images to disk for analysis

        Returns:
            dict: Comprehensive metrics including timing, throughput, and counts
        """
        collection_name = f"{self.dataset_name}_{condition_name}_{uuid.uuid4().hex[:8]}"

        print(f"\n{'='*60}")
        print(f"Running {condition_name} benchmark for {self.dataset_name}")
        print(f"Store enabled: {store_enabled}")
        print(f"Collection: {collection_name}")
        temp_path = f"/raid/jioffe/tmp/store_task/{self.run_id}_{collection_name[:20]}..."
        print(f"Temp files will be saved to: {temp_path}")

        # Log extraction configuration for consistency verification
        print("📋 EXTRACTION CONFIG:")
        print("  extract_text=True")
        print("  extract_tables=True")
        print("  extract_charts=True")
        print("  extract_images=False")  # Consistent across conditions
        print("  extract_infographics=False")
        print("  extract_page_as_image=False")

        if store_enabled and store_params:
            print("📦 STORE CONFIG:")
            print(f"  structured={store_params.get('structured', 'N/A')}")
            print(f"  images={store_params.get('images', 'N/A')}")
            params = store_params.get("params", {})
            print(f"  endpoint={params.get('endpoint', 'N/A')}")
            print(f"  collection_name={params.get('collection_name', 'N/A')}")

        print(f"{'='*60}")

        # Timing and metrics
        metrics = {
            "condition": condition_name,
            "dataset": self.dataset_name,
            "store_enabled": store_enabled,
            "collection_name": collection_name,
            "file_count": self.file_count,
            "page_count": self.page_count,
            "model_name": self.model_name,
            "dense_dim": self.dense_dim,
            "timestamp": datetime.now().isoformat(),
        }

        # Create ingestor
        ingestor = self.create_ingestor_base(collection_name, store_enabled, store_params)

        # Run ingestion
        ingestion_start = time.time()
        try:
            print(f"⚡ Starting ingestion at {time.strftime('%H:%M:%S')}...")
            results, failures = ingestor.ingest(show_progress=True, return_failures=True, save_to_disk=True)
            ingestion_end = time.time()

            metrics["ingestion_time"] = ingestion_end - ingestion_start
            print(f"⚡ Ingestion completed at {time.strftime('%H:%M:%S')} (duration: {metrics['ingestion_time']:.2f}s)")
            metrics["ingestion_success"] = True
            metrics["document_count"] = len(results)  # Number of documents processed
            metrics["failure_count"] = len(failures)
            metrics["failures"] = failures
            metrics["pages_per_sec"] = self.page_count / metrics["ingestion_time"]
            metrics["files_per_sec"] = self.file_count / metrics["ingestion_time"]

            print(
                f"Ingestion completed: {metrics['document_count']} documents, " f"{metrics['failure_count']} failures"
            )
            print(f"Ingestion time: {metrics['ingestion_time']:.2f}s")
            print(f"Throughput: {metrics['pages_per_sec']:.2f} pages/s, " f"{metrics['files_per_sec']:.2f} files/s")

            # Manual flush for immediate consistency
            flush_start = time.time()
            try:
                from pymilvus import Collection, connections, MilvusClient

                connections.connect("default", host=self.hostname, port="19530")
                collection = Collection(collection_name)
                collection.load()

                collection.flush()
                client = MilvusClient(f"http://{self.hostname}:19530")
                client.refresh_load(collection_name)

                flush_time = time.time() - flush_start
                final_row_count = collection.num_entities

                print(f"✅ Manual flush completed: {final_row_count} rows in {flush_time:.2f}s")
                metrics["flush_time"] = flush_time
                metrics["final_row_count"] = final_row_count

            except Exception as e:
                print(f"⚠️  Flush failed: {e}")
                metrics["flush_time"] = 0
                metrics["final_row_count"] = 0
                metrics["flush_error"] = str(e)

            # Detailed chunk analysis using segment_results like bo20_e2e.py
            try:
                # Get total element count from LazyLoadedList lengths
                total_elements = sum(len(doc_results) for doc_results in results)
                metrics["total_elements"] = total_elements

                # Get detailed chunk counts by type using segment_results
                text_results, table_results, chart_results = segment_results(results)
                text_chunks = sum([len(chunk) for chunk in text_results])
                table_chunks = sum([len(chunk) for chunk in table_results])
                chart_chunks = sum([len(chunk) for chunk in chart_results])

                metrics["text_chunks"] = text_chunks
                metrics["table_chunks"] = table_chunks
                metrics["chart_chunks"] = chart_chunks

                print(f"Total elements: {total_elements}")
                print(f"Chunk breakdown - Text: {text_chunks}, " f"Tables: {table_chunks}, Charts: {chart_chunks}")

                # Analyze images for store task conditions
                if store_enabled and not skip_image_analysis:
                    try:
                        # Save images to disk for detailed analysis
                        image_output_dir = f"/raid/jioffe/tmp/store_task/{self.run_id}_{collection_name}_images"

                        print("Analyzing images from store task...")
                        image_disk_start = time.time()
                        image_counts = save_images_from_ingestor_results(
                            results,
                            image_output_dir,
                            save_charts=True,
                            save_tables=True,
                            save_infographics=True,
                            save_page_images=True,
                            save_raw_images=True,
                            organize_by_type=True,
                        )
                        image_disk_time = time.time() - image_disk_start

                        # Add image metrics to benchmark results
                        metrics["images_saved_to_disk"] = image_counts.get("total", 0)
                        metrics["chart_images"] = image_counts.get("chart", 0)
                        metrics["table_images"] = image_counts.get("table", 0)
                        metrics["infographic_images"] = image_counts.get("infographic", 0)
                        metrics["page_images"] = image_counts.get("page_image", 0)
                        metrics["raw_images"] = image_counts.get("image", 0)
                        metrics["image_output_dir"] = image_output_dir
                        metrics["image_disk_save_time"] = image_disk_time

                        print(f"Images saved to disk: {image_counts.get('total', 0)} in {image_disk_time:.2f}s")
                        for img_type, count in image_counts.items():
                            if img_type != "total" and count > 0:
                                print(f"  - {img_type}: {count}")
                        print(f"Image files saved to: {image_output_dir}")
                        print(
                            f"Image disk save rate: {image_counts.get('total', 0) / image_disk_time:.1f} images/sec"
                            if image_disk_time > 0
                            else ""
                        )

                    except Exception as img_e:
                        print(f"Warning: Image analysis failed: {img_e}")
                        metrics["image_analysis_error"] = str(img_e)

            except Exception as e:
                print(f"Note: Could not get detailed chunk counts: {e}")
                metrics["total_elements"] = "N/A"
                metrics["text_chunks"] = "N/A"
                metrics["table_chunks"] = "N/A"
                metrics["chart_chunks"] = "N/A"

            # Check Milvus statistics
            try:
                milvus_uri = f"http://{self.hostname}:19530"
                milvus_chunks(milvus_uri, collection_name)
            except Exception as e:
                print(f"Warning: Could not get Milvus stats: {e}")

            # Test retrieval
            retrieval_start = time.time()
            try:
                test_queries = [
                    "What is the main topic discussed?",
                    "What financial information is provided?",
                    "What are the key findings?",
                ]

                query_results = nvingest_retrieval(
                    test_queries,
                    collection_name,
                    hybrid=True,
                    embedding_endpoint=f"http://{self.hostname}:8012/v1",
                    embedding_model_name=self.model_name,
                    model_name=self.model_name,
                    top_k=5,
                    gpu_search=False,
                )

                metrics["retrieval_time"] = time.time() - retrieval_start
                metrics["retrieval_success"] = True
                metrics["query_count"] = len(test_queries)

                # Check for MinIO URLs in results if store was enabled
                if store_enabled:
                    urls_found = self.check_minio_urls(query_results)
                    metrics["minio_urls_found"] = urls_found
                    print(f"MinIO URLs found in results: {urls_found}")

                print(f"Retrieval time: {metrics['retrieval_time']:.2f}s")

            except Exception as e:
                print(f"Retrieval failed: {e}")
                metrics["retrieval_time"] = None
                metrics["retrieval_success"] = False
                metrics["retrieval_error"] = str(e)

        except Exception as e:
            print(f"Ingestion failed: {e}")
            metrics["ingestion_success"] = False
            metrics["ingestion_error"] = str(e)
            return metrics

        # Store validation if enabled
        if store_enabled:
            try:
                minio_stats = self.validate_minio_storage()
                metrics.update(minio_stats)
            except Exception as e:
                print(f"MinIO validation failed: {e}")
                metrics["minio_validation_error"] = str(e)

        # Clean up temporary files after analysis
        try:
            import shutil

            temp_dir = f"/raid/jioffe/tmp/store_task/{self.run_id}_{collection_name}"
            image_dir = f"/raid/jioffe/tmp/store_task/{self.run_id}_{collection_name}_images"

            # Clean up main temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")

            # Clean up image analysis directory
            if os.path.exists(image_dir):
                shutil.rmtree(image_dir)
                print(f"Cleaned up image directory: {image_dir}")

        except Exception as e:
            print(f"Warning: Could not clean up temp directories: {e}")

        self.results[condition_name] = metrics
        return metrics

    def check_minio_urls(self, query_results):
        """
        Verify that MinIO URLs are present in query results when store is enabled.

        Args:
            query_results: Results from nvingest_retrieval query

        Returns:
            int: Count of MinIO URLs found in the results
        """
        url_count = 0
        for result_set in query_results:
            for result in result_set:
                # Handle the correct Milvus result structure
                metadata = None

                # Check if result has 'entity' field (new structure)
                if "entity" in result and isinstance(result["entity"], dict):
                    entity = result["entity"]
                    if "content_metadata" in entity:
                        metadata = entity["content_metadata"]
                # Fallback to old structure
                elif "metadata" in result:
                    metadata = result["metadata"]
                elif "content_metadata" in result:
                    metadata = result["content_metadata"]

                if metadata:
                    # Parse JSON if it's a string
                    if isinstance(metadata, str):
                        import json

                        try:
                            metadata = json.loads(metadata)
                        except (ValueError, json.JSONDecodeError):
                            pass

                    if isinstance(metadata, dict):
                        # Check content_url field
                        if metadata.get("content_url"):
                            url_count += 1

                        # Check various metadata fields for uploaded_image_url
                        for meta_key in ["image_metadata", "table_metadata", "chart_metadata"]:
                            if meta_key in metadata and isinstance(metadata[meta_key], dict):
                                sub_metadata = metadata[meta_key]
                                if "uploaded_image_url" in sub_metadata and sub_metadata["uploaded_image_url"]:
                                    url_count += 1

                        # Check legacy source_metadata structure
                        source_metadata = metadata.get("source_metadata", {})
                        if isinstance(source_metadata, dict):
                            if "source_location" in source_metadata and source_metadata["source_location"]:
                                url_count += 1
                            if "uploaded_image_url" in source_metadata and source_metadata["uploaded_image_url"]:
                                url_count += 1

        return url_count

    def validate_minio_storage(self):
        """
        Validate MinIO storage accessibility and health.

        Returns:
            dict: MinIO validation metrics including accessibility status
        """
        import requests

        minio_stats = {"minio_accessible": False, "minio_object_count": 0, "sample_url_reachable": False}

        try:
            # Try to access MinIO health endpoint
            response = requests.get(f"http://{self.hostname}:9000/minio/health/live", timeout=5)
            minio_stats["minio_accessible"] = response.status_code == 200
        except (ConnectionError, TimeoutError, requests.RequestException):
            pass

        # Additional MinIO validation can be added here:
        # - List bucket contents for object count
        # - Verify sample URLs are accessible
        # - Check bucket permissions

        return minio_stats

    def save_results(self, output_dir="test_results"):
        """
        Save comprehensive benchmark results to JSON and CSV files.

        Args:
            output_dir (str): Directory to save result files

        Returns:
            tuple: Paths to (detailed_json, summary_csv) files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        results_file = f"{output_dir}/store_benchmark_{self.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save summary CSV
        summary_file = f"{output_dir}/store_benchmark_{self.run_id}_summary.csv"
        self.create_summary_csv(summary_file)

        print("\nResults saved to:")
        print(f"  Detailed: {results_file}")
        print(f"  Summary: {summary_file}")

        return results_file, summary_file

    def create_summary_csv(self, filename):
        """
        Create summary CSV file with key metrics for easy analysis and comparison.

        Args:
            filename (str): Path to output CSV file
        """
        import csv

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            headers = [
                "condition",
                "dataset",
                "store_enabled",
                "ingestion_time",
                "flush_time",
                "final_row_count",
                "pages_per_sec",
                "files_per_sec",
                "document_count",
                "total_elements",
                "failure_count",
                "text_chunks",
                "table_chunks",
                "chart_chunks",
                "images_saved_to_disk",
                "chart_images",
                "table_images",
                "infographic_images",
                "page_images",
                "raw_images",
                "image_disk_save_time",
                "retrieval_time",
                "minio_urls_found",
                "ingestion_success",
                "retrieval_success",
            ]
            writer.writerow(headers)

            # Data rows
            for condition, metrics in self.results.items():
                row = [
                    metrics.get("condition", ""),
                    metrics.get("dataset", ""),
                    metrics.get("store_enabled", False),
                    metrics.get("ingestion_time", ""),
                    metrics.get("flush_time", ""),
                    metrics.get("final_row_count", ""),
                    metrics.get("pages_per_sec", ""),
                    metrics.get("files_per_sec", ""),
                    metrics.get("document_count", ""),
                    metrics.get("total_elements", ""),
                    metrics.get("failure_count", ""),
                    metrics.get("text_chunks", ""),
                    metrics.get("table_chunks", ""),
                    metrics.get("chart_chunks", ""),
                    metrics.get("images_saved_to_disk", ""),
                    metrics.get("chart_images", ""),
                    metrics.get("table_images", ""),
                    metrics.get("infographic_images", ""),
                    metrics.get("page_images", ""),
                    metrics.get("raw_images", ""),
                    metrics.get("image_disk_save_time", ""),
                    metrics.get("retrieval_time", ""),
                    metrics.get("minio_urls_found", ""),
                    metrics.get("ingestion_success", ""),
                    metrics.get("retrieval_success", ""),
                ]
                writer.writerow(row)


def main():
    """
    Main function to run store task benchmarking.

    Supports benchmarking both baseline (store OFF) and store-enabled conditions
    with comprehensive performance metrics and image analysis.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Benchmark nv-ingest store task performance impact",
        epilog="Example: python store_benchmark.py bo20 --hostname localhost",
    )
    parser.add_argument("dataset", choices=["bo20", "bo767"], help="Dataset to benchmark")
    parser.add_argument("--hostname", default="localhost", help="Service hostname")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline (store OFF)")
    parser.add_argument("--store-only", action="store_true", help="Run only store ON condition")
    parser.add_argument("--skip-image-analysis", action="store_true", help="Skip saving images to disk for analysis")
    parser.add_argument("--output-dir", default="test_results/store_task", help="Output directory for results")

    args = parser.parse_args()

    # Determine dataset path
    dataset_path = f"/raid/{os.environ.get('USER', '')}/{args.dataset}/"
    if not os.path.exists(dataset_path):
        dataset_path = f"/datasets/{args.dataset}/"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return 1

    print(f"Using dataset path: {dataset_path}")

    # Create benchmark instance
    benchmark = StoreBenchmark(args.dataset, dataset_path, args.hostname)

    # MinIO connection parameters
    store_params = {
        "structured": True,
        "images": True,
        "params": {
            "access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "endpoint": "minio:9000",  # Use Docker service name for server connections
            "bucket_name": os.getenv("MINIO_BUCKET", "nv-ingest"),
            "secure": False,
            "region": "us-east-1",
        },
    }

    try:
        # Run baseline (store OFF) unless store-only
        if not args.store_only:
            benchmark.run_benchmark(
                "baseline_store_off", store_enabled=False, skip_image_analysis=args.skip_image_analysis
            )

        # Run store ON unless baseline-only
        if not args.baseline_only:
            benchmark.run_benchmark(
                "store_on", store_enabled=True, store_params=store_params, skip_image_analysis=args.skip_image_analysis
            )

        # Save results
        benchmark.save_results(args.output_dir)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        for condition, metrics in benchmark.results.items():
            print(f"\n{condition}:")
            print(f"  Ingestion time: {metrics.get('ingestion_time', 'N/A'):.2f}s")
            print(f"  Flush time: {metrics.get('flush_time', 'N/A'):.2f}s")
            print(f"  Final row count: {metrics.get('final_row_count', 'N/A')}")
            print(f"  Throughput: {metrics.get('pages_per_sec', 'N/A'):.2f} pages/s")
            print(f"  Documents: {metrics.get('document_count', 'N/A')}")
            print(f"  Total elements: {metrics.get('total_elements', 'N/A')}")
            print(f"  Text chunks: {metrics.get('text_chunks', 'N/A')}")
            print(f"  Table chunks: {metrics.get('table_chunks', 'N/A')}")
            print(f"  Chart chunks: {metrics.get('chart_chunks', 'N/A')}")
            print(f"  Failures: {metrics.get('failure_count', 'N/A')}")
            if metrics.get("store_enabled"):
                print(f"  MinIO URLs found: {metrics.get('minio_urls_found', 'N/A')}")
                # Show image analysis results
                total_images = metrics.get("images_saved_to_disk", 0)
                if total_images > 0:
                    image_disk_time = metrics.get("image_disk_save_time", 0)
                    print(f"  Images saved to disk: {total_images}")
                    print(f"    - Chart images: {metrics.get('chart_images', 0)}")
                    print(f"    - Table images: {metrics.get('table_images', 0)}")
                    print(f"    - Infographic images: {metrics.get('infographic_images', 0)}")
                    print(f"    - Page images: {metrics.get('page_images', 0)}")
                    print(f"    - Raw images: {metrics.get('raw_images', 0)}")
                    if image_disk_time > 0:
                        rate = total_images / image_disk_time
                        print(f"    - Disk save time: {image_disk_time:.2f}s ({rate:.1f} images/sec)")
                    else:
                        print(f"    - Disk save time: {image_disk_time:.2f}s")
                else:
                    print("  Images saved to disk: 0")

        return 0

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
