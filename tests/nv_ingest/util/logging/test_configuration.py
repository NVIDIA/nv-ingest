# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from nv_ingest.util.logging.configuration import LogLevel
from nv_ingest.util.logging.configuration import configure_logging


def test_valid_log_levels(caplog):
    """
    Test that the logger is correctly configured for each valid log level.
    """
    for level in LogLevel:
        logger = logging.getLogger("test")
        configure_logging(logger, level.value)
        logger.log(getattr(logging, level.value), f"Test message for {level.value}")
        assert logger.level == getattr(logging, level.value), f"Logger level not set correctly for {level.value}"
        assert caplog.record_tuples[-1][1] == getattr(
            logging, level.value
        ), f"Logging output level incorrect for {level.value}"
        caplog.clear()


def test_invalid_log_level():
    """
    Test that configuring the logger with an invalid log level raises ValueError.
    """
    logger = logging.getLogger("test_invalid")
    with pytest.raises(ValueError):
        configure_logging(logger, "INVALID_LEVEL")


@pytest.fixture
def logger():
    """
    Fixture to create a logger instance.
    """
    return logging.getLogger("test_logger")


def test_logger_level_set_correctly(logger):
    """
    Test that the logger level is set correctly when a valid level is provided.
    """
    for level in LogLevel:
        configure_logging(logger, level.value)
        assert logger.level == getattr(logging, level.value), "Logger level did not match expected level"


def test_logger_raises_for_invalid_level(logger):
    """
    Test that the logger raises a ValueError for invalid log levels.
    """
    with pytest.raises(ValueError) as exc_info:
        configure_logging(logger, "NO_LEVEL")
    assert "Invalid log level: NO_LEVEL" in str(exc_info.value), "Expected specific error message for invalid log level"
