# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import (
    PipelineCreationSchema,
    start_pipeline_subprocess,
)
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(logger, local_log_level)


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

        # The main program will exit, and the atexit handler will terminate the subprocess group

    except Exception as e:
        logger.error(f"Error running pipeline subprocess or ingestion: {e}")

        # The atexit handler will ensure subprocess termination
        sys.exit(1)


if __name__ == "__main__":
    main()
