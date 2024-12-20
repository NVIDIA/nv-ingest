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
        .embed()
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
        # Collect environment parameters, falling back to schema defaults if not set
        config_data = {
            "cached_grpc_endpoint": os.getenv("CACHED_GRPC_ENDPOINT"),
            "cached_infer_protocol": os.getenv("CACHED_INFER_PROTOCOL"),
            "embedding_nim_endpoint": os.getenv("EMBEDDING_NIM_ENDPOINT"),
            "deplot_http_endpoint": os.getenv("DEPLOT_HTTP_ENDPOINT"),
            "deplot_infer_protocol": os.getenv("DEPLOT_INFER_PROTOCOL"),
            "ingest_log_level": os.getenv("INGEST_LOG_LEVEL"),
            "message_client_host": os.getenv("MESSAGE_CLIENT_HOST"),
            "message_client_port": os.getenv("MESSAGE_CLIENT_PORT"),
            "message_client_type": os.getenv("MESSAGE_CLIENT_TYPE"),
            "minio_bucket": os.getenv("MINIO_BUCKET"),
            "mrc_ignore_numa_check": os.getenv("MRC_IGNORE_NUMA_CHECK"),
            "otel_exporter_otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "paddle_grpc_endpoint": os.getenv("PADDLE_GRPC_ENDPOINT"),
            "paddle_http_endpoint": os.getenv("PADDLE_HTTP_ENDPOINT"),
            "paddle_infer_protocol": os.getenv("PADDLE_INFER_PROTOCOL"),
            "redis_morpheus_task_queue": os.getenv("REDIS_MORPHEUS_TASK_QUEUE"),
            "yolox_infer_protocol": os.getenv("YOLOX_INFER_PROTOCOL"),
            "yolox_grpc_endpoint": os.getenv("YOLOX_GRPC_ENDPOINT"),
            "vlm_caption_endpoint": os.getenv("VLM_CAPTION_ENDPOINT"),
        }

        # Filter out None values to let the schema defaults handle them
        config_data = {key: value for key, value in config_data.items() if value is not None}

        # Construct the pipeline configuration
        config = PipelineCreationSchema(**config_data)

        # Start the pipeline subprocess
        pipeline_process = start_pipeline_subprocess(config)

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
