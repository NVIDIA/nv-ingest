# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import logging.config
from enum import Enum


class LogLevel(str, Enum):
    DEFAULT = "DEFAULT"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def configure_logging(level_name: str) -> None:
    """
    Configures global logging.

    Parameters
    ----------
    level_name : str
        The name of the logging level (e.g., "DEBUG", "INFO").
    """
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    # Use dictConfig with disable_existing_loggers=True to eliminate any import-time
    # handlers from third-party libraries and ensure a single, consistent console handler.
    config_dict = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": numeric_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "level": numeric_level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config_dict)
