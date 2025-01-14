import logging
import os
import sys
import time

from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging

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
            extract_images=False,
        )
        .split(
            split_by="word",
            split_length=300,
            split_overlap=10,
            max_character_length=5000,
            sentence_window_size=0,
        )
        .embed(text=True, tables=True)
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

        # Run ingestion after starting the pipeline
        run_ingestor()

        pipeline_process.wait()

        # The main program will exit, and the atexit handler will terminate the subprocess group

    except Exception as e:
        logger.error(f"Error running pipeline subprocess or ingestion: {e}")

        # The atexit handler will ensure subprocess termination
        sys.exit(1)


if __name__ == "__main__":
    main()
