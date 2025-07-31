# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
import sys

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(local_log_level)


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
            extract_infographics=False,
            text_depth="page",
        )
        .split(chunk_size=1024, chunk_overlap=150)
        .embed()
    )

    try:
        results, _ = ingestor.ingest(show_progress=False, return_failures=True)
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")
    print(f"Got {len(results)} results.")


def main():
    """
    Launch the libmode pipeline service and run the ingestor against it.
    Uses the embedded default libmode pipeline configuration.
    """
    try:
        pipeline = run_pipeline(
            block=False,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        time.sleep(10)
        run_ingestor()
        # Run other code...
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
    finally:
        pipeline.stop()
        logger.info("Shutting down pipeline...")


if __name__ == "__main__":
    main()
