import logging
import os
import sys
import time

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import (
    PipelineCreationSchema,
    start_pipeline_subprocess,
)
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging

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
    client = NvIngestClient(
        message_client_allocator=SimpleClient, message_client_port=7671, message_client_hostname="localhost"
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
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
        .vdb_upload(
            collection_name="test",
            milvus_uri="milvus.db",
            sparse=False,
            dense_dim=2048,
        )
    )

    try:
        _ = ingestor.ingest()
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")


def main():
    try:
        # Possibly override config parameters
        config_data = {}

        # Filter out None values to let the schema defaults handle them
        config_data = {key: value for key, value in config_data.items() if value is not None}

        # Construct the pipeline configuration
        config = PipelineCreationSchema(**config_data)

        # Start the pipeline subprocess
        pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)

        # Optionally, wait a bit before starting the ingestor to ensure the pipeline is ready
        time.sleep(10)

        # If the pipeline is not running, exit immediately
        if pipeline_process.poll() is not None:
            raise RuntimeError("Error running pipeline subprocess.")

        # Run ingestion after starting the pipeline
        run_ingestor()

        # The main program will exit, and the atexit handler will terminate the subprocess group

    except Exception as e:
        logger.error(f"Error running pipeline subprocess or ingestion: {e}")

        # The atexit handler will ensure subprocess termination
        sys.exit(1)


if __name__ == "__main__":
    main()
