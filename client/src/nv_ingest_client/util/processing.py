# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent
import json
import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from click import style
from nv_ingest_client.util.util import check_ingest_result
from pydantic import BaseModel
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class IngestJobFailure(Exception):
    """Custom exception to handle failed job ingestion results."""

    def __init__(self, message: str, description: str, annotations: Dict[str, Any]):
        super().__init__(message)
        self.description = description
        self.annotations = annotations


def handle_future_result(
    future: concurrent.futures.Future,
    timeout=10,
) -> Tuple[Dict[str, Any], str]:
    """
    Handle the result of a completed future job and process annotations.

    This function processes the result of a future, extracts annotations (if any), logs them,
    and checks the validity of the ingest result. If the result indicates a failure, a
    `RuntimeError` is raised with a description of the failure.

    Parameters
    ----------
    future : concurrent.futures.Future
        A future object representing an asynchronous job. The result of this job will be
        processed once it completes.
    timeout : Optional[int], default=None
        Maximum time to wait for the future result before timing out.

    Returns
    -------
    Tuple[Dict[str, Any], str]
        - The result of the job as a dictionary, after processing and validation.
        - The trace_id returned by the submission endpoint

    Raises
    ------
    IngestJobFailure
        If the job result is invalid, this exception is raised with the failure description
        and the full result for further inspection.
    Exception
        For all other unexpected errors.

    Notes
    -----
    - The `future.result()` is assumed to return a tuple where the first element is the actual
      result (as a dictionary), and the second element (if present) can be ignored.
    - Annotations in the result (if any) are logged for debugging purposes.
    - The `check_ingest_result` function (assumed to be defined elsewhere) is used to validate
      the result. If the result is invalid, a `RuntimeError` is raised.

    Examples
    --------
    Suppose we have a future object representing a job, a dictionary of futures to job IDs,
    and a directory for saving results:

    >>> future = concurrent.futures.Future()
    >>> result, trace_id = handle_future_result(future, timeout=60)

    In this example, the function processes the completed job and returns the result dictionary.
    If the job fails, it raises a `RuntimeError`.

    See Also
    --------
    check_ingest_result : Function to validate the result of the job.
    """

    try:
        result, _, trace_id = future.result(timeout=timeout)[0]
        if ("annotations" in result) and result["annotations"]:
            annotations = result["annotations"]
            for key, value in annotations.items():
                logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

        failed, description = check_ingest_result(result)

        if failed:
            raise IngestJobFailure(f"Ingest job failed: {description}", description, result.get("annotations"))
    except Exception as e:
        raise e

    return (result, trace_id)


def highlight_error_in_original(original_str: str, task_name: str, error_detail: Dict[str, Any]) -> str:
    """
    Highlights the error-causing text in the original JSON string based on the error type.

    This function identifies the problematic portion of the JSON string by inspecting the
    provided error details. For errors due to extra fields, it highlights the extra field
    (using blue and bold formatting). For errors due to missing fields or insufficient
    string length, it appends a clear message indicating the missing field and its location.

    Parameters
    ----------
    original_str : str
        The original JSON string that caused the error. This string will be modified to highlight
        the problematic field.
    task_name : str
        The name of the task associated with the error. This is used in the error message when
        highlighting missing fields.
    error_detail : Dict[str, Any]
        A dictionary containing details about the error. Expected keys are:
        - 'type' (str): The type of error (e.g., "value_error.extra", "value_error.missing",
          "value_error.any_str.min_length").
        - 'loc' (List[Any]): A list representing the path to the error-causing field in the JSON structure.

    Returns
    -------
    str
        The modified JSON string with the error-causing field highlighted or a message appended indicating
        the missing field.

    Notes
    -----
    - The function uses the `style` function to apply formatting to the error-causing text.
    - If the error detail does not include the expected keys, a fallback is used and the original string is returned.
    """
    try:
        error_type: str = error_detail.get("type", "unknown")
        loc: List[Any] = error_detail.get("loc", [])
        if loc:
            # Build a string representation of the error location
            error_location: str = "->".join(map(str, loc))
            error_key: str = str(loc[-1])
        else:
            error_location = "unknown"
            error_key = ""

        if error_type == "value_error.extra" and error_key:
            # Use regex substitution to only highlight the first occurrence of the error field.
            highlighted_key = style(error_key, fg="blue", bold=True)
            highlighted_str = re.sub(f'("{re.escape(error_key)}")', highlighted_key, original_str, count=1)
        elif error_type in ["value_error.missing", "value_error.any_str.min_length"]:
            # Provide a clear message about the missing field.
            if error_key:
                missing_field = style(f"'{error_key}'", fg="blue", bold=True)
            else:
                missing_field = style(f"at '{error_location}'", fg="blue", bold=True)
            highlighted_str = (
                f"{original_str}\n(Schema Error): Missing required parameter for task '{task_name}': {missing_field}"
            )
        else:
            # For any other error types, attempt to highlight the field if available.
            if error_key:
                highlighted_key = style(error_key, fg="blue", bold=True)
                highlighted_str = re.sub(f'("{re.escape(error_key)}")', highlighted_key, original_str, count=1)
            else:
                highlighted_str = original_str
    except Exception as e:
        logger.error(f"Error while highlighting error in original string: {e}")
        highlighted_str = original_str

    return highlighted_str


def format_validation_error(e: ValidationError, task_id: str, original_str: str) -> str:
    """
    Formats validation errors with appropriate highlights and returns a detailed error message.

    Parameters
    ----------
    e : ValidationError
        The validation error raised by the schema.
    task_id : str
        The identifier of the task for which the error occurred.
    original_str : str
        The original JSON string that caused the validation error.

    Returns
    -------
    str
        A detailed error message with highlighted problematic fields.
    """
    error_messages: List[str] = []
    for error in e.errors():
        error_message = f"(Schema Error): {error['msg']}"
        highlighted_str = highlight_error_in_original(original_str, task_id, error)
        error_messages.append(f"{error_message}\n -> {highlighted_str}")
    return "\n".join(error_messages)


def check_schema(schema: Type[BaseModel], options: dict, task_id: str, original_str: str) -> BaseModel:
    """
    Validates the provided options against the given schema and returns a schema instance.

    Parameters
    ----------
    schema : Type[BaseModel]
        A Pydantic model class used for validating the options.
    options : dict
        The options dictionary to validate.
    task_id : str
        The identifier of the task associated with the options.
    original_str : str
        The original JSON string representation of the options.

    Returns
    -------
    BaseModel
        An instance of the validated schema populated with the provided options.

    Raises
    ------
    ValueError
        If validation fails, a ValueError is raised with a formatted error message highlighting
        the problematic parts of the original JSON string.
    """
    try:
        return schema(**options)
    except ValidationError as e:
        error_message = format_validation_error(e, task_id, original_str)
        raise ValueError(error_message) from e
