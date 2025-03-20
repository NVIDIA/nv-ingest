# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any
from typing import Callable
from typing import Dict

from langdetect.lang_detect_exception import LangDetectException

from nv_ingest_api.internal.enums.common import LanguageEnum

logger = logging.getLogger(__name__)


def langdetect_exception_handler(func: Callable, **kwargs: Dict[str, Any]) -> Callable:
    """
    A decorator that handles `LangDetectException` for language detection functions.

    This decorator wraps a function that performs language detection and catches any `LangDetectException` that occurs
    during its execution.
    If such an exception is raised, it logs a warning and returns a default value of `LanguageEnum.UNKNOWN`.

    Parameters
    ----------
    func : callable
        The function to be decorated. This function is expected to handle language detection.

    kwargs : dict
        Additional keyword arguments to be passed to the function.

    Returns
    -------
    callable
        The wrapped function that executes `func` with exception handling.

    Notes
    -----
    If a `LangDetectException` is raised while executing the wrapped function, the exception is logged,
    and `LanguageEnum.UNKNOWN` is returned as a fallback value.

    Examples
    --------
    >>> @langdetect_exception_handler
    ... def detect_language(text):
    ...     # Function implementation here
    ...     pass
    ...
    >>> detect_language('This is a test sentence.')
    <LanguageEnum.EN: 'en'>

    If a `LangDetectException` is encountered, the function will return `LanguageEnum.UNKNOWN`:

    >>> detect_language('')
    <LanguageEnum.UNKNOWN: 'unknown'>

    Raises
    ------
    LangDetectException
        The exception raised by the wrapped function is caught and handled by logging a warning
        and returning `LanguageEnum.UNKNOWN`.
    """

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except LangDetectException as e:
            log_error_message = f"LangDetectException: {e}"
            logger.warning(log_error_message)
            return LanguageEnum.UNKNOWN

    return inner_function
