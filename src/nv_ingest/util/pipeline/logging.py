# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from morpheus.utils.logger import configure_logging

from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest.util.pipeline.stage_builders import *

# Convert log level from string to logging level
_log_level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_log_level(str_level):
    """
    Converts the log level from a string to a logging level.
    """
    return _log_level_mapping.get(str_level.upper(), logging.INFO)


def setup_logging(log_level):
    """
    Configures logging based on the provided log level or the INGEST_LOG_LEVEL environment variable.
    """
    # Check for INGEST_LOG_LEVEL environment variable
    env_log_level = os.getenv("INGEST_LOG_LEVEL", log_level)
    if env_log_level:
        log_level = env_log_level
        if log_level in ("DEFAULT",):
            log_level = "INFO"

    log_level_value = _log_level_mapping.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_value, format="%(asctime)s - %(levelname)s - %(message)s")
    configure_logging(log_level=log_level_value)
    configure_local_logging(logger, log_level_value)
