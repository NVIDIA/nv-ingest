# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import functools
import inspect
import re
from typing import Any, Optional, Callable, Tuple
from functools import wraps

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.logging import TaskResultStatus, annotate_task_result
from nv_ingest_api.util.control_message.validators import cm_ensure_payload_not_null, cm_set_failure

logger = logging.getLogger(__name__)


def nv_ingest_node_failure_try_except(  # New name to distinguish
    annotation_id: Optional[str] = None,
    payload_can_be_empty: bool = False,
    raise_on_failure: bool = False,
    skip_processing_if_failed: bool = True,
    forward_func: Optional[Callable[[Any], Any]] = None,
) -> Callable:
    """
    Decorator that wraps function execution in a try/except block to handle
    failures by annotating an IngestControlMessage. Replaces the context
    manager approach for potentially simpler interaction with frameworks like Ray.

    Parameters
    ----------
    annotation_id : Optional[str]
        A unique identifier for annotation. If None, attempts to auto-detect
        from the stage instance's stage_name property.
    payload_can_be_empty : bool, optional
        If False, the message payload must not be null.
    raise_on_failure : bool, optional
        If True, exceptions are raised; otherwise, they are annotated.
    skip_processing_if_failed : bool, optional
        If True, skip processing if the message is already marked as failed.
    forward_func : Optional[Callable[[Any], Any]]
        If provided, a function to forward the message when processing is skipped.
    """

    def extract_message_and_prefix(args: Tuple) -> Tuple[Any, Tuple]:
        """Extracts control_message and potential 'self' prefix."""
        # (Keep the implementation from the original decorator)
        if args and hasattr(args[0], "get_metadata"):
            return args[0], ()
        elif len(args) >= 2 and hasattr(args[1], "get_metadata"):
            return args[1], (args[0],)
        else:
            # Be more specific in error if possible
            arg_types = [type(arg).__name__ for arg in args]
            raise ValueError(f"No IngestControlMessage found in first or second argument. Got types: {arg_types}")

    def decorator(func: Callable) -> Callable:
        func_name = func.__name__  # Get function name for logging/errors

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug(f"sync_wrapper for {func_name}: Entering.")

            # Determine the annotation_id to use
            resolved_annotation_id = annotation_id

            # If no explicit annotation_id provided, try to get it from self.stage_name
            if resolved_annotation_id is None and len(args) >= 1:
                stage_instance = args[0]  # 'self' in method calls
                if hasattr(stage_instance, "stage_name") and stage_instance.stage_name:
                    resolved_annotation_id = stage_instance.stage_name
                    logger.debug("Using auto-detected annotation_id from stage_name: " f"'{resolved_annotation_id}'")
                else:
                    # Fallback to function name if no stage_name available
                    resolved_annotation_id = func_name
                    logger.debug(
                        "No stage_name available, using function name as annotation_id: " f"'{resolved_annotation_id}'"
                    )
            elif resolved_annotation_id is None:
                # Fallback to function name if no annotation_id and no instance
                resolved_annotation_id = func_name
                logger.debug(
                    "No annotation_id provided and no instance available, using function name: "
                    f"'{resolved_annotation_id}'"
                )

            try:
                control_message, prefix = extract_message_and_prefix(args)
            except ValueError as e:
                logger.error(f"sync_wrapper for {func_name}: Failed to extract control message. Error: {e}")
                raise

            # --- Skip logic ---
            is_failed = control_message.get_metadata("cm_failed", False)
            if is_failed and skip_processing_if_failed:
                logger.warning(f"sync_wrapper for {func_name}: Skipping processing, message already marked failed.")
                if forward_func:
                    logger.debug("sync_wrapper: Forwarding skipped message.")
                    return forward_func(control_message)  # Assume forward_func is sync here
                else:
                    logger.debug("sync_wrapper: Returning skipped message as is.")
                    return control_message

            # --- Main execution block ---
            result = None
            try:
                # Payload check
                if not payload_can_be_empty:
                    cm_ensure_payload_not_null(control_message)

                # Rebuild args and call original sync function
                new_args = prefix + (control_message,) + args[len(prefix) + 1 :]
                logger.debug(f"sync_wrapper for {func_name}: Calling func...")
                result = func(*new_args, **kwargs)
                logger.debug(f"sync_wrapper for {func_name}: func call completed.")

                # Success annotation
                logger.debug(f"sync_wrapper for {func_name}: Annotating success.")
                annotate_task_result(
                    control_message=result if result is not None else control_message,
                    # Annotate result or original message
                    result=TaskResultStatus.SUCCESS,
                    task_id=resolved_annotation_id,
                )
                logger.debug(f"sync_wrapper for {func_name}: Success annotation done. Returning result.")
                return result

            except Exception as e:
                # --- Failure Handling ---
                error_message = f"Error in {func_name}: {e}"
                logger.error(f"sync_wrapper for {func_name}: Caught exception: {error_message}", exc_info=True)

                # Annotate failure on the original message object
                try:
                    cm_set_failure(control_message, error_message)
                    annotate_task_result(
                        control_message=control_message,
                        result=TaskResultStatus.FAILURE,
                        task_id=resolved_annotation_id,
                        message=error_message,
                    )
                    logger.debug(f"sync_wrapper for {func_name}: Failure annotation complete.")
                except Exception as anno_err:
                    logger.exception(
                        f"sync_wrapper for {func_name}: CRITICAL - Error during failure annotation: {anno_err}"
                    )

                # Decide whether to raise or return annotated message
                if raise_on_failure:
                    logger.debug(f"sync_wrapper for {func_name}: Re-raising exception as configured.")
                    raise e  # Re-raise the original exception
                else:
                    logger.debug(
                        f"sync_wrapper for {func_name}: Suppressing exception and returning annotated message."
                    )
                    # Return the original control_message, now annotated with failure
                    return control_message

        return sync_wrapper

    return decorator


def nv_ingest_node_failure_context_manager(
    annotation_id: str,
    payload_can_be_empty: bool = False,
    raise_on_failure: bool = False,
    skip_processing_if_failed: bool = True,
    forward_func: Optional[Callable[[Any], Any]] = None,
) -> Callable:
    """
    Decorator that applies a failure context manager around a function processing an IngestControlMessage.
    Works with both synchronous and asynchronous functions, and supports class methods (with 'self').

    Parameters
    ----------
    annotation_id : str
        A unique identifier for annotation.
    payload_can_be_empty : bool, optional
        If False, the message payload must not be null.
    raise_on_failure : bool, optional
        If True, exceptions are raised; otherwise, they are annotated.
    skip_processing_if_failed : bool, optional
        If True, skip processing if the message is already marked as failed.
    forward_func : Optional[Callable[[Any], Any]]
        If provided, a function to forward the message when processing is skipped.

    Returns
    -------
    Callable
        The decorated function.
    """

    def extract_message_and_prefix(args: Tuple) -> Tuple[Any, Tuple]:
        """
        Determines if the function is a method (first argument is self) or a standalone function.
        Returns a tuple (control_message, prefix) where prefix is a tuple of preceding arguments to be preserved.
        """
        if args and hasattr(args[0], "get_metadata"):
            # Standalone function: first argument is the message.
            return args[0], ()
        elif len(args) >= 2 and hasattr(args[1], "get_metadata"):
            # Method: first argument is self, second is the message.
            return args[1], (args[0],)
        else:
            raise ValueError("No IngestControlMessage found in the first or second argument.")

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                control_message, prefix = extract_message_and_prefix(args)
                is_failed = control_message.get_metadata("cm_failed", False)
                if not is_failed or not skip_processing_if_failed:
                    ctx_mgr = CMNVIngestFailureContextManager(
                        control_message=control_message,
                        annotation_id=annotation_id,
                        raise_on_failure=raise_on_failure,
                        func_name=func.__name__,
                    )
                    try:
                        ctx_mgr.__enter__()
                        if not payload_can_be_empty:
                            cm_ensure_payload_not_null(control_message)
                        # Rebuild argument list preserving any prefix (e.g. self).
                        new_args = prefix + (ctx_mgr.control_message,) + args[len(prefix) + 1 :]
                        result = await func(*new_args, **kwargs)
                    except Exception as e:
                        ctx_mgr.__exit__(type(e), e, e.__traceback__)
                        raise
                    else:
                        ctx_mgr.__exit__(None, None, None)
                        return result
                else:
                    if forward_func:
                        return await forward_func(control_message)
                    else:
                        return control_message

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                control_message, prefix = extract_message_and_prefix(args)
                is_failed = control_message.get_metadata("cm_failed", False)
                if not is_failed or not skip_processing_if_failed:
                    with CMNVIngestFailureContextManager(
                        control_message=control_message,
                        annotation_id=annotation_id,
                        raise_on_failure=raise_on_failure,
                        func_name=func.__name__,
                    ) as ctx_mgr:
                        if not payload_can_be_empty:
                            cm_ensure_payload_not_null(control_message)
                        new_args = prefix + (ctx_mgr.control_message,) + args[len(prefix) + 1 :]
                        return func(*new_args, **kwargs)
                else:
                    if forward_func:
                        return forward_func(control_message)
                    else:
                        return control_message

            return sync_wrapper

    return decorator


def nv_ingest_source_failure_context_manager(
    annotation_id: str,
    payload_can_be_empty: bool = False,
    raise_on_failure: bool = False,
) -> Callable:
    """
    A decorator that ensures any function's output is treated as a IngestControlMessage for annotation.
    It applies a context manager to handle success and failure annotations based on the function's execution.

    Parameters
    ----------
    annotation_id : str
        Unique identifier used for annotating the function's output.
    payload_can_be_empty : bool, optional
        Specifies if the function's output IngestControlMessage payload can be empty, default is False.
    raise_on_failure : bool, optional
        Determines if an exception should be raised upon function failure, default is False.

    Returns
    -------
    Callable
        A decorator that ensures function output is processed for success or failure annotation.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> IngestControlMessage:
            try:
                result = func(*args, **kwargs)
                if not isinstance(result, IngestControlMessage):
                    raise TypeError(f"{func.__name__} output is not a IngestControlMessage as expected.")
                if not payload_can_be_empty and result.get_metadata("payload") is None:
                    raise ValueError(f"{func.__name__} IngestControlMessage payload cannot be null.")

                # Success annotation.
                annotate_task_result(result, result=TaskResultStatus.SUCCESS, task_id=annotation_id)
            except Exception as e:
                error_message = f"Error in {func.__name__}: {e}"
                # Prepare a new IngestControlMessage for failure annotation if needed.
                if "result" not in locals() or not isinstance(result, IngestControlMessage):
                    result = IngestControlMessage()
                cm_set_failure(result, error_message)
                annotate_task_result(
                    result,
                    result=TaskResultStatus.FAILURE,
                    task_id=annotation_id,
                    message=error_message,
                )
                if raise_on_failure:
                    raise
            return result

        return wrapper

    return decorator


class CMNVIngestFailureContextManager:
    """
    Context manager for handling IngestControlMessage failures during processing, providing
    a structured way to annotate and manage failures and successes.

    Parameters
    ----------
    control_message : IngestControlMessage
        The IngestControlMessage instance to be managed.
    annotation_id : str
        The task's unique identifier for annotation purposes.
    raise_on_failure : bool, optional
        Determines whether to raise an exception upon failure. Defaults to False, which
        means failures are annotated rather than raising exceptions.
    func_name : str, optional
        The name of the function being wrapped, used to annotate error messages uniformly.
        If None, stack introspection is used to deduce a likely function name. Defaults to None.

    Returns
    -------
    None
    """

    def __init__(
        self,
        control_message: IngestControlMessage,
        annotation_id: str,
        raise_on_failure: bool = False,
        func_name: str = None,
    ):
        self.control_message = control_message
        self.annotation_id = annotation_id
        self.raise_on_failure = raise_on_failure
        if func_name is not None:
            self._func_name = func_name
        else:
            try:
                # Use stack introspection to get a candidate function name.
                stack = inspect.stack()
                # Use the third frame as a heuristic; adjust if needed.
                candidate = stack[2].function if len(stack) > 2 else "UnknownFunction"
                # Remove any whitespace and limit the length to 50 characters.
                candidate = re.sub(r"\s+", "", candidate)[:50]
                self._func_name = candidate if candidate else "UnknownFunction"
            except Exception:
                self._func_name = "UnknownFunction"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:  # An exception occurred
            error_message = f"Error in {self._func_name}: {exc_value}"
            if self.control_message is not None:
                cm_set_failure(self.control_message, error_message)
                annotate_task_result(
                    self.control_message,
                    result=TaskResultStatus.FAILURE,
                    task_id=self.annotation_id,
                    message=error_message,
                )
            # Propagate the exception if raise_on_failure is True; otherwise, suppress it.
            if self.raise_on_failure:
                return False
            return True

        annotate_task_result(
            self.control_message,
            result=TaskResultStatus.SUCCESS,
            task_id=self.annotation_id,
        )
        return False


def unified_exception_handler(func):
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                err_msg = f"{func_name}: error: {e}"
                logger.exception(err_msg, exc_info=True)
                raise type(e)(err_msg) from e

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_name = func.__name__
                err_msg = f"{func_name}: error: {e}"
                logger.exception(err_msg, exc_info=True)
                raise type(e)(err_msg) from e

        return sync_wrapper
