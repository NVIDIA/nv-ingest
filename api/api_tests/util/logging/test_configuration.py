# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from nv_ingest_api.util.logging.configuration import LogLevel
from nv_ingest_api.util.logging.configuration import configure_logging


@pytest.mark.parametrize(
    "level_name,expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_configure_logging_valid_levels(level_name, expected_level, capsys):
    """
    Test that `configure_logging` correctly configures the global logger for valid levels.

    Purpose
    -------
    Verify that when `configure_logging` is called with a valid log level string,
    the logger outputs messages at that level to `stdout`.

    Method
    ------
    - Parametrize the test with various valid levels.
    - Call `configure_logging` with the level.
    - Log a message at the expected level.
    - Use `capsys` to capture output and assert the message is present in stdout.
    """
    configure_logging(level_name)
    logger = logging.getLogger("test_logger")
    logger.log(expected_level, "Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out
    assert level_name in captured.out


def test_configure_logging_invalid_level():
    """
    Test that `configure_logging` raises ValueError when provided an invalid level string.

    Purpose
    -------
    Ensure the function validates the input level string strictly and raises an appropriate error.

    Method
    ------
    - Pass an invalid log level string.
    - Assert that a `ValueError` is raised and the error message matches expectations.
    """
    with pytest.raises(ValueError, match="Invalid log level: FAKELEVEL"):
        configure_logging("FAKELEVEL")


def test_configure_logging_case_insensitive(capsys):
    """
    Test that `configure_logging` accepts log levels in a case-insensitive manner.

    Purpose
    -------
    Confirm that even if the input string is in lowercase, the function still configures correctly.

    Method
    ------
    - Call `configure_logging` with 'debug' (lowercase).
    - Log a DEBUG message.
    - Capture stdout and assert the message is present.
    """
    configure_logging("debug")  # lower case should still work
    logger = logging.getLogger("test_logger")
    logger.debug("Case insensitive test")
    captured = capsys.readouterr()
    assert "Case insensitive test" in captured.out


@pytest.mark.parametrize("log_level", list(LogLevel))
def test_loglevel_enum_values(log_level):
    """
    Test that `LogLevel` enum contains expected string values.

    Purpose
    -------
    Verify that all enum members are strings and their names are reflected in their values.

    Method
    ------
    - Iterate over all members of `LogLevel`.
    - Assert that each value is a string.
    - Assert that the name of the enum is part of its value or the value is 'DEFAULT'.
    """
    assert isinstance(log_level.value, str)
    assert log_level.name in log_level.value or log_level.value == "DEFAULT"


def test_configure_logging_force_reconfigures(capsys):
    """
    Test that `configure_logging` can reconfigure the logger when called multiple times.

    Purpose
    -------
    Ensure that `force=True` in `logging.basicConfig` allows the logging configuration
    to be reset and updated even if it was already configured.

    Method
    ------
    - First configure logging at 'WARNING' level.
    - Log a DEBUG message (should not appear).
    - Reconfigure logging at 'DEBUG' level.
    - Log a DEBUG message again (should now appear).
    - Assert outputs accordingly after each phase.
    """
    configure_logging("WARNING")
    logger = logging.getLogger("test_logger")
    logger.debug("Should not appear")
    captured = capsys.readouterr()
    assert "Should not appear" not in captured.out

    configure_logging("DEBUG")
    logger.debug("Should appear now")
    captured = capsys.readouterr()
    assert "Should appear now" in captured.out
