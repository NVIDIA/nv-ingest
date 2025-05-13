import logging
import os
import time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import PipelineCreationSchema

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
        # .split()
        # .embed()
    )

    try:
        results = ingestor.ingest()
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")
    print(ingest_json_results_to_blob(results[0]))


def main():
    # Possibly override config parameters
    config_data = {}

    # Filter out None values to let the schema defaults handle them
    config_data = {key: value for key, value in config_data.items() if value is not None}

    # Construct the pipeline configuration
    ingest_config = PipelineCreationSchema(**config_data)

    pipeline = None
    try:
        pipeline, _ = run_pipeline(ingest_config, block=False)
        time.sleep(10)
        run_ingestor()
        # Run other code...
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    finally:
        logger.info("Shutting down pipeline...")
        if pipeline:
            pipeline.stop()


if __name__ == "__main__":
    main()
