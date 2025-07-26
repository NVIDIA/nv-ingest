# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging

# Configure the logger
logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "DEFAULT")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"

configure_local_logging(local_log_level)


def main():
    """
    Launch the libmode pipeline service using the embedded default configuration.
    """
    try:
        _ = run_pipeline(
            block=True,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
            stderr=sys.stderr,
            stdout=sys.stdout,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
