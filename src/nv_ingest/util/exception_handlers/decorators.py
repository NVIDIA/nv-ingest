# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import typing
from functools import wraps

from morpheus.messages import ControlMessage
from morpheus.utils.control_message_utils import cm_ensure_payload_not_null
from morpheus.utils.control_message_utils import cm_set_failure

from nv_ingest.util.tracing.logging import TaskResultStatus
from nv_ingest.util.tracing.logging import annotate_task_result


def nv_ingest_node_failure_context_manager(
    annotation_id: str,
    payload_can_be_empty: bool = False,
    raise_on_failure: bool = False,
    skip_processing_if_failed: bool = True,
    forward_func=None,
) -> typing.Callable:
    """
    A decorator that applies a default failure context manager around a function to manage
    the execution and potential failure of operations involving ControlMessages.

    Parameters
    ----------
    annotation_id : str
        A unique identifier used for annotating the task's result.
    payload_can_be_empty : bool, optional
        If False, the payload of the ControlMessage will be checked to ensure it's not null,
        raising an exception if it is null. Defaults to False, enforcing payload presence.
    raise_on_failure : bool, optional
        If True, an exception is raised if the decorated function encounters an error.
        Otherwise, the error is handled silently by annotating the ControlMessage. Defaults to False.
    skip_processing_if_failed:
        If True, skips the processing of the decorated function if the control message has already
        been marked as failed. If False, the function will be processed regardless of the failure
        status of the ControlMessage. Defaults to True.

    Returns
    -------
    Callable
        A decorator that wraps the given function with failure handling logic.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(control_message: ControlMessage, *args, **kwargs):
            # Quick return if the ControlMessage has already failed
            is_failed = control_message.get_metadata("cm_failed", False)
            if not is_failed or not skip_processing_if_failed:
                with CMNVIngestFailureContextManager(
                    control_message=control_message,
                    annotation_id=annotation_id,
                    raise_on_failure=raise_on_failure,
                ) as ctx_mgr:
                    if not payload_can_be_empty:
                        cm_ensure_payload_not_null(control_message=control_message)

                    control_message = func(ctx_mgr.control_message, *args, **kwargs)

            else:
                if forward_func:
                    control_message = forward_func(control_message)

            return control_message

        return wrapper

    return decorator


def nv_ingest_source_failure_context_manager(
    annotation_id: str,
    payload_can_be_empty: bool = False,
    raise_on_failure: bool = False,
) -> typing.Callable:
    """
    A decorator that ensures any function's output is treated as a ControlMessage for annotation.
    It applies a context manager to handle success and failure annotations based on the function's execution.

    Parameters
    ----------
    annotation_id : str
        Unique identifier used for annotating the function's output.
    payload_can_be_empty : bool, optional
        Specifies if the function's output ControlMessage payload can be empty, default is False.
    raise_on_failure : bool, optional
        Determines if an exception should be raised upon function failure, default is False.

    Returns
    -------
    Callable
        A decorator that ensures function output is processed for success or failure annotation.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> ControlMessage:
            # Attempt to execute the decorated function and process its output.
            try:
                result = func(*args, **kwargs)
                if not isinstance(result, ControlMessage):
                    raise TypeError("Function output is not a ControlMessage as expected.")
                if not payload_can_be_empty and result.get_metadata("payload") is None:
                    raise ValueError("ControlMessage payload cannot be null.")

                # Success annotation.
                annotate_task_result(result, result=TaskResultStatus.SUCCESS, task_id=annotation_id)
            except Exception as e:
                # Prepare a new ControlMessage for failure annotation if needed.
                result = (
                    ControlMessage() if "result" not in locals() or not isinstance(result, ControlMessage) else result
                )
                cm_set_failure(result, str(e))
                annotate_task_result(
                    result,
                    result=TaskResultStatus.FAILURE,
                    task_id=annotation_id,
                    message=str(e),
                )
                if raise_on_failure:
                    raise
            return result

        return wrapper

    return decorator


class CMNVIngestFailureContextManager:
    """
    Context manager for handling ControlMessage failures during processing, providing
    a structured way to annotate and manage failures and successes.

    Parameters
    ----------
    control_message : ControlMessage
        The ControlMessage instance to be managed.
    annotation_id : str
        The task's unique identifier for annotation purposes.
    raise_on_failure : bool, optional
        Determines whether to raise an exception upon failure. Defaults to False, which
        means failures are annotated rather than raising exceptions.

    Returns
    -------
    None

    """

    def __init__(
        self,
        control_message: ControlMessage,
        annotation_id: str,
        raise_on_failure: bool = False,
    ):
        self.control_message = control_message
        self.raise_on_failure = raise_on_failure
        self.annotation_id = annotation_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:  # An exception occurred
            if self.raise_on_failure:
                raise

            if self.control_message is not None:
                cm_set_failure(self.control_message, str(exc_value))
                annotate_task_result(
                    self.control_message,
                    result=TaskResultStatus.FAILURE,
                    task_id=self.annotation_id,
                    message=str(exc_value),
                )

            return True  # Indicate that we handled the exception

        annotate_task_result(
            self.control_message,
            result=TaskResultStatus.SUCCESS,
            task_id=self.annotation_id,
        )

        return False  # Indicate that we did not handle the exception
