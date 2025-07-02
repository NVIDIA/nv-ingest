import argparse
import logging
import os
import time

from nv_ingest_api.util.logging.configuration import configure_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

# Configure logging to see output from the client
configure_logging("INFO")
logger = logging.getLogger(__name__)

def run_benchmark(input_path: str, caption: bool):
    """
    Runs an ingestion pipeline on the given input path, with an option to enable image captioning,
    and benchmarks the performance.

    :param input_path: Path to the document or directory to ingest.
    :param caption: Boolean flag to enable/disable the .caption() stage.
    """
    print(f"--- Running Ingestion Benchmark ---")
    print(f"Input: {input_path}")
    print(f"Captioning Enabled: {caption}")
    print("---------------------------------")

    # This client connects to the nv-ingest-ms-runtime service in your docker-compose setup.
    # The default port 7671 is for the SimpleBroker.
    client = NvIngestClient(
        message_client_allocator=RedisClient,
        message_client_port=6379,
        message_client_hostname="localhost"
    )
    time.sleep(5)

    # Begin building the ingestion pipeline
    ingestor = Ingestor(client=client)

    # Use .directory() or .files() depending on the input path
    if os.path.isdir(input_path):
        ingestor = ingestor.directory(input_path)
    else:
        ingestor = ingestor.files(input_path)

    # Add the standard extraction task for text and images
    ingestor = ingestor.extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True
    )

    # Conditionally add the VLM captioning task
    if caption:
        ingestor = ingestor.caption()

    # --- Execute the pipeline and time it ---
    logger.info("Starting ingestion...This may take a moment.")
    t0 = time.time()
    results = ingestor.ingest(show_progress=True)
    t1 = time.time()
    elapsed_time = t1 - t0
    print("---------------------------------")
    print(f"Ingestion finished. Total time taken: {elapsed_time:.2f} seconds")

    # --- Analyze results for captioned images ---
    if caption and results:
        captioned_image_count = 0
        # results is a list of results, one per document
        for doc_result in results:
            # each doc_result is a list of chunks/entities
            for entity in doc_result:
                # Check if the entity is an image and has a caption field
                if entity.get("type") == "image" and entity.get("caption"):
                    captioned_image_count += 1
        print(f"Found {captioned_image_count} captioned images in the output.")

    print("---------------------------------")
    return elapsed_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark nv-ingest captioning performance.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/multimodal_test.pdf",
        help="Path to the document file or directory to ingest. Defaults to data/multimodal_test.pdf."
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Enable the VLM captioning stage in the pipeline."
    )
    args = parser.parse_args()

    run_benchmark(args.input_path, args.caption) 