# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(local_log_level)


def main():
    # Possibly override config parameters
    config_data = {}

    # Filter out None values to let the schema defaults handle them
    config_data = {key: value for key, value in config_data.items() if value is not None}

    # Construct the pipeline configuration
    ingest_config = PipelineCreationSchema(**config_data)

    try:
        _ = run_pipeline(ingest_config, block=True)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
