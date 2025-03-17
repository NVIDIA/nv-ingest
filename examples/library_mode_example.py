import logging
import os
import sys
import time

from nv_ingest_client.client import Ingestor
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess

# from dotenv import load_dotenv

# load_dotenv(".env")


# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(logger, local_log_level)


def run_ingestor():
    """
    Set up and run the ingestion process to send traffic against the pipeline.
    """
    logger.info("Setting up Ingestor client...")

    ingestor = (
        Ingestor(
            message_client_allocator=SimpleClient,
            message_client_port=7671,
            message_client_hostname=os.getenv("MESSAGE_CLIENT_HOSTNAME", "localhost"),
        )
        .files("data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_infographics=True,
            text_depth="page",
        )
        .embed()
        .caption()
        # .vdb_upload(
        #   collection_name=collection_name,
        #   milvus_uri=milvus_uri,
        #   sparse=sparse,
        #   # for llama-3.2 embedder, use 1024 for e5-v5
        #   dense_dim=2048
        # )
    )

    try:
        logger.info("Starting ingestion..")
        t0 = time.time()
        results = ingestor.ingest()
        t1 = time.time()
        logger.info("Ingestion completed successfully.")
        logger.info(f"Time taken: {t1-t0} seconds")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    # results blob is directly inspectable
    print(ingest_json_results_to_blob(results[0]))


def main():
    # Construct the pipeline configuration
    config = PipelineCreationSchema()

    # Start the pipeline subprocess
    pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)

    # Optionally, wait a bit before starting the ingestor to ensure the pipeline is ready
    time.sleep(10)

    if pipeline_process.poll() is not None:
        logger.error("Error running pipeline subprocess.")
        sys.exit(1)

    # Run ingestion if the pipeline is alive
    run_ingestor()

    # The main program will exit, and the atexit handler will terminate the subprocess group


if __name__ == "__main__":
    main()
