import logging
import sys
import time

from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient

# Configure the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def run_ingestor():
    """
    Set up and run the ingestion process to send traffic against the pipeline.
    """
    logger.info("Setting up Ingestor client...")
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost"
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
        ).split(
            split_by="word",
            split_length=300,
            split_overlap=10,
            max_character_length=5000,
            sentence_window_size=0,
        )
    )

    try:
        results = ingestor.ingest()
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")


def main():
    try:
        # Start the pipeline subprocess
        pipeline_process = start_pipeline_subprocess()

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
