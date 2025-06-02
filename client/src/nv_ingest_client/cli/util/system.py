# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging


def configure_logging(logger, log_level: str):
    """
    Configures the logging level based on a log_level string.

    Parameters
    ----------
    logger: logging.Logger
        The logger to configure.
    log_level : str
        The logging level as a string, expected to be one of
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Convert the log level string to a logging level.
    numeric_level = level_dict.get(log_level.upper(), None)
    if numeric_level is None:
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure the logger to the specified level.
    logging.basicConfig(level=numeric_level)
    logger.setLevel(numeric_level)
    logger.debug(f"Logging configured to {log_level} level.")
