# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime  # Import the timezone class directly
from datetime import timezone

import pytest

from nv_ingest.util.converters.datetools import datetimefrompdfmeta
from nv_ingest.util.converters.datetools import remove_tz
from nv_ingest.util.converters.datetools import validate_iso8601


@pytest.mark.parametrize(
    "pdf_date, keep_tz, expected_result",
    [
        # ("D:20211222141131-07'00'", False, "2021-12-22T21:11:31"),  # PDF date with timezone, removed
        (
            "D:20211222141131-07'00'",
            True,
            "2021-12-22T14:11:31-07:00",
        ),  # PDF date with timezone, kept
        ("D:20211222141131", False, "2021-12-22T14:11:31"),  # PDF date without timezone
        ("Not a date", False, None),  # Malformed date, expecting exception handling
    ],
)
def test_datetimefrompdfmeta(pdf_date, keep_tz, expected_result):
    if expected_result:
        assert datetimefrompdfmeta(pdf_date, keep_tz) == expected_result
    else:
        datetimefrompdfmeta(pdf_date, keep_tz)


def test_remove_tz():
    datetime_with_tz = datetime(2021, 12, 22, 14, 11, 31, tzinfo=timezone.utc)
    expected_result = datetime(2021, 12, 22, 14, 11, 31)
    assert remove_tz(datetime_with_tz).isoformat() == expected_result.isoformat()

    datetime_without_tz = datetime(2021, 12, 22, 14, 11, 31)
    assert remove_tz(datetime_without_tz).isoformat() == datetime_without_tz.isoformat()


@pytest.mark.parametrize(
    "date_string, is_valid",
    [
        ("2021-12-22T21:11:31", True),
        ("2021-12-22", True),
        ("This is not a date", False),
    ],
)
def test_validate_iso8601(date_string, is_valid):
    if is_valid:
        validate_iso8601(date_string)
    else:
        with pytest.raises(ValueError):
            validate_iso8601(date_string)
