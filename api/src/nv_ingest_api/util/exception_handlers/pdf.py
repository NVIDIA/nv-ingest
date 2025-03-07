# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from nv_ingest_api.internal.enums.common import StatusEnum, TaskTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata

logger = logging.getLogger(__name__)


def pdfium_exception_handler(descriptor):
    """
    A decorator that handles exceptions for functions interacting with PDFium.

    This decorator wraps a function and catches any exceptions that occur during its execution.
    If an exception is raised, it logs a warning with a descriptor and the function name,
    then returns an empty list as a fallback value.

    Parameters
    ----------
    descriptor : str
        A string descriptor to identify the context or source of the function being wrapped.
        This descriptor is included in the log message if an exception occurs.

    Returns
    -------
    callable
        A decorator function that wraps the target function with exception handling.

    Notes
    -----
    This decorator is useful for ensuring that functions interacting with PDFium can gracefully handle errors
    without interrupting the entire processing pipeline.

    Examples
    --------
    >>> @pdfium_exception_handler("PDF Processing")
    ... def process_pdf(file_path):
    ...     # Function implementation here
    ...     pass
    ...
    >>> process_pdf("example.pdf")
    []

    Raises
    ------
    Exception
        Any exception raised by the wrapped function is caught, logged, and handled by returning an empty list.
    """

    def outer_function(func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error_message = f"{descriptor}:{func.__name__} error:{e}"
                logger.warning(log_error_message)
                return []

        return inner_function

    return outer_function


def create_exception_tag(error_message, source_id=None):
    """
    Creates a metadata tag for logging or reporting an exception.

    This function generates a metadata dictionary containing information about the exception,
    including the task type, status, source identifier, and error message.
    The metadata is validated and returned as a list containing a single entry.

    Parameters
    ----------
    error_message : str
        The error message describing the exception.
    source_id : Optional[str], default=None
        The identifier for the source related to the error, if available.

    Returns
    -------
    list
        A list containing a single entry, which is a tuple. The first element of the tuple is `None`,
        and the second element is the validated metadata dictionary as a `dict`.

    Notes
    -----
    This function is typically used to generate error metadata for tracking and logging purposes.

    Examples
    --------
    >>> create_exception_tag("File not found", source_id="12345")
    [[None, {'task': 'EXTRACT', 'status': 'ERROR', 'source_id': '12345', 'error_msg': 'File not found'}]]

    Raises
    ------
    ValidationError
        If the metadata does not pass validation.
    """
    unified_metadata = {}

    error_metadata = {
        "task": TaskTypeEnum.EXTRACT,
        "status": StatusEnum.ERROR,
        "source_id": source_id,
        "error_msg": error_message,
    }

    unified_metadata["error_metadata"] = error_metadata

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [[None, validated_unified_metadata.model_dump()]]
