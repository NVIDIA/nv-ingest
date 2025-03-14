# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime
from datetime import timezone

from dateutil.parser import parse

from nv_ingest_api.util.exception_handlers.converters import datetools_exception_handler


@datetools_exception_handler
def datetimefrompdfmeta(pdf_formated_date: str, keep_tz: bool = False) -> str:
    """
    Convert PDF metadata formatted date string to a datetime object.

    Parameters
    ----------
    pdf_formated_date : str
        A date string in standard PDF metadata format.
        Example: `str("D:20211222141131-07'00'")`
    keep_tz : bool, optional
        Keep or remove the timezone attribute of the parsed datetime object. If `False` (necessary for arrow format),
        the timezone offset will be added to the datetime. Parsed datetimes will be in the same local time.

    Returns
    -------
    str
        A datetime object parsed from the input date string in ISO 8601 format.

    """

    try:
        # standard pdf date format
        pattern = "D:%Y%m%d%H%M%S%z"
        # clean up date string
        cleaned_date_string = pdf_formated_date[:-1].replace("'", ":")
        parsed_dt_tz = datetime.strptime(cleaned_date_string, pattern)
    except ValueError:
        parsed_dt_tz = parse(pdf_formated_date, fuzzy=True)

    if not keep_tz:
        return remove_tz(parsed_dt_tz).isoformat()

    return parsed_dt_tz.isoformat()


def remove_tz(datetime_obj: datetime) -> datetime:
    """
    Remove timezone and add offset to a datetime object.

    Parameters
    ----------
    datetime_obj : datetime.datetime
        A datetime object with or without the timezone attribute set.

    Returns
    -------
    datetime.datetime
        A datetime object with the timezone offset added and the timezone attribute removed.

    """

    if datetime_obj.tzinfo is not None:  # If timezone info is present
        # Convert to UTC
        datetime_obj = datetime_obj.astimezone(timezone.utc)
        # Remove timezone information
        datetime_obj = datetime_obj.replace(tzinfo=None)

    return datetime_obj


def validate_iso8601(date_string: str) -> None:
    """
    Verify that the given date string is in ISO 8601 format.

    Parameters
    ----------
    date_string : str
        A date string in human-readable format, ideally ISO 8601.

    Raises
    ------
    ValueError
        If the date string is not in a valid ISO 8601 format.
    """

    assert datetime.fromisoformat(date_string)
