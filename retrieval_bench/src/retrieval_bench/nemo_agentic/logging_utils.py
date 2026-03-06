# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from BatsResearch/trove (Apache-2.0):
# https://github.com/BatsResearch/trove/blob/main/src/trove/logging_utils.py
"""Simple utilities to create python loggers."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from rich.logging import RichHandler


@dataclass
class LoggingModuleConfig:
    LOGGING_DISABLE: bool = False
    LOG_LEVEL: str = "INFO"
    LOGGING_DISABLE_RICH: bool = False


config = LoggingModuleConfig(
    LOG_LEVEL=os.environ.get("LOG_LEVEL", "INFO"),
    LOGGING_DISABLE=os.environ.get("LOGGING_DISABLE", "false").lower() == "true",
    LOGGING_DISABLE_RICH=os.environ.get("LOGGING_DISABLE_RICH", "false").lower() == "true",
)


# LOGGING_FORMAT_STR_RICH = "%(funcName)s() %(message)s"
LOGGING_FORMAT_STR_RICH = "%(message)s"
LOGGING_FORMAT_STR = "%(asctime)s %(module)s:%(lineno)s, [%(funcName)s] (%(levelname)s)- %(message)s"
LOGGING_TIME_FORMAT_STR = "%H:%M:%S"

STR_LOG_LEVEL_TO_INT = {
    "CRITICAL": logging.CRITICAL,
    "DEBUG": logging.DEBUG,
    "ERROR": logging.ERROR,
    "FATAL": logging.FATAL,
    "INFO": logging.INFO,
    "NOTSET": logging.NOTSET,
    "WARN": logging.WARN,
    "WARNING": logging.WARNING,
}
INT_LOG_LEVEL_TO_STR = {v: k for k, v in STR_LOG_LEVEL_TO_INT.items()}


def ensure_str_log_level(level: Union[str, int]) -> str:
    """Ensures log level is valid and converts it to ``str`` levels recognized by python logging."""
    if isinstance(level, str):
        if level.upper() not in STR_LOG_LEVEL_TO_INT:
            raise ValueError(f"Log level can only be one of '{list(STR_LOG_LEVEL_TO_INT.keys())}'. Got '{level}'")
        return level.upper()
    if isinstance(level, int):
        if level not in STR_LOG_LEVEL_TO_INT.values():
            raise ValueError(f"Log level can only be one of '{list(INT_LOG_LEVEL_TO_STR.keys())}'. Got '{level}'")
        return INT_LOG_LEVEL_TO_STR[level]
    raise TypeError(f"Log level can only be 'int' or 'str'. Got '{type(level)}'")


class LoggerConfig:
    def __init__(self, level: Union[int, str]) -> None:
        self.level = ensure_str_log_level(level)

    def is_debug(self) -> bool:
        return STR_LOG_LEVEL_TO_INT[self.level] == logging.DEBUG

    def is_info(self) -> bool:
        return STR_LOG_LEVEL_TO_INT[self.level] == logging.INFO

    def is_warning(self) -> bool:
        return STR_LOG_LEVEL_TO_INT[self.level] == logging.WARNING


class LoggerWrapper:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def info(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.error(*args, **kwargs)

    def log(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.log(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if not config.LOGGING_DISABLE:
            return self._logger.critical(*args, **kwargs)


def rpath(path: Union[Path, str, os.PathLike]) -> str:
    """Make sure path starts with ``/`` or ``./`` so rich will highlight it in logs."""
    path = Path(path).as_posix()
    if not (path.startswith("/") or path.startswith(".")):
        path = "./" + path
    return path


def get_logger_with_config(
    name: Optional[str] = "ART",
    log_level: Optional[str] = None,
    rank: Optional[int] = None,
    force: bool = False,
) -> Tuple[Union[logging.Logger, LoggerWrapper], LoggerConfig]:
    """Creates and returns a logger instance and a config object."""
    logger = logging.getLogger(name)
    log_level = ensure_str_log_level(config.LOG_LEVEL if log_level is None else log_level)
    if force and len(logger.handlers):
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    if not len(logger.handlers):
        logger.setLevel(getattr(logging, log_level))
        if config.LOGGING_DISABLE_RICH:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, log_level))
            format_str = LOGGING_FORMAT_STR
            if rank is not None:
                format_str = format_str.replace("%(asctime)s", f"%(asctime)s <R{rank}>")
            if name:
                format_str = f"[{name}] " + format_str
            handler.setFormatter(logging.Formatter(format_str, LOGGING_TIME_FORMAT_STR))
        else:
            handler = RichHandler(
                level=getattr(logging, log_level),
                omit_repeated_times=False,
                locals_max_length=None,
                locals_max_string=None,
            )
            handler.setLevel(getattr(logging, log_level))
            format_str = LOGGING_FORMAT_STR_RICH
            if rank is not None:
                format_str = f"<R{rank}> " + LOGGING_FORMAT_STR_RICH
            if name:
                format_str = f"[{name}] " + format_str
            handler.setFormatter(logging.Formatter(format_str, LOGGING_TIME_FORMAT_STR))
        logger.addHandler(handler)
    logging_config = LoggerConfig(level=log_level)
    return logger, logging_config
