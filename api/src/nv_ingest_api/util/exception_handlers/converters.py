# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Callable
from typing import Dict

from nv_ingest_api.util.converters import datetools

logger = logging.getLogger(__name__)


def datetools_exception_handler(func: Callable, **kwargs: Dict[str, Any]) -> Callable:
    """
    A decorator that handles exceptions for date-related functions.

    This decorator wraps a function that processes dates and catches any exceptions that occur during its execution.
    If an exception is raised, it logs a warning and returns the current UTC time as an ISO 8601 formatted string.

    Parameters
    ----------
    func : Callable
        The function to be decorated. This function is expected to handle date operations.

    kwargs : dict
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    Callable
        The wrapped function that executes `func` with exception handling.

    Notes
    -----
    If an exception is raised while executing the wrapped function, the current UTC time (with timezone information
    removed)
    will be returned as an ISO 8601 formatted string.

    Examples
    --------
    >>> @datetools_exception_handler
    ... def parse_date(date_str):
    ...     return datetime.strptime(date_str, '%Y-%m-%d')
    ...
    >>> parse_date('2024-08-22')
    datetime.datetime(2024, 8, 22, 0, 0)

    If the input is invalid, the current UTC time without timezone information is returned:

    >>> parse_date('invalid-date')
    '2024-08-22T12:34:56'

    Raises
    ------
    Exception
        Any exception raised by the wrapped function is caught, logged, and handled by returning the current UTC time.
    """

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error_message = f"Invalid date format: {e}"
            logger.debug(log_error_message)
            return datetools.remove_tz(datetime.now(timezone.utc)).isoformat()

    return inner_function
