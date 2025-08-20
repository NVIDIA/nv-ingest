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

    # Scorched-earth reset: remove ALL existing handlers from root and named loggers
    # to ensure there is exactly one handler after configuration.
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # Clear handlers from all known loggers and make them propagate to root
    for name, logger_obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger_obj, logging.Logger):
            for h in list(logger_obj.handlers):
                logger_obj.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            # Ensure messages bubble to root; levels will be controlled centrally
            logger_obj.propagate = True
            logger_obj.setLevel(logging.NOTSET)

    # Use dictConfig to establish a single console handler on the root logger.
    config_dict = {
        "version": 1,
        # We already cleared handlers above; keep loggers enabled so they propagate to root
        "disable_existing_loggers": False,
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

    # Enforce exactly one handler remains attached to root (keep first StreamHandler)
    root_logger = logging.getLogger()
    if len(root_logger.handlers) > 1:
        keep = None
        for h in list(root_logger.handlers):
            if keep is None and isinstance(h, logging.StreamHandler):
                keep = h
                continue
            root_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # Route warnings module through logging
    try:
        import logging as _logging

        _logging.captureWarnings(True)
    except Exception:
        pass
