# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from nv_ingest.util.converters.datetools import datetools_exception_handler
from nv_ingest.util.converters.datetools import remove_tz


# Example functions to test the decorator
@datetools_exception_handler
def test_func_raises_exception():
    raise ValueError("Test exception")


@datetools_exception_handler
def test_func_success():
    return "Success"


def test_datetools_exception_handler_with_exception():
    """
    Test the decorator with a function that raises an exception,
    checking that the returned date is within a few minutes of the current time.
    """
    start_time = remove_tz(datetime.now(timezone.utc))

    result = test_func_raises_exception()

    # Convert result back to datetime object for comparison
    result_datetime = datetime.fromisoformat(result)

    end_time = remove_tz(datetime.now(timezone.utc))

    # Check the result is within a reasonable time delta (e.g., a few minutes)
    time_delta = timedelta(minutes=5)

    assert (
        start_time - time_delta
    ) <= result_datetime, "The returned datetime should be within a few minutes of the current time"
    assert result_datetime <= (
        end_time + time_delta
    ), "The returned datetime should be within a few minutes of the current time"


def test_datetools_exception_handler_without_exception():
    """
    Test the decorator with a function that does not raise an exception.
    """
    result = test_func_success()
    assert result == "Success", "Decorator should not interfere with the function's normal execution"
