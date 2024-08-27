# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
from datetime import datetime


def traceable(trace_name=None):
    """
    A decorator that adds entry and exit trace timestamps to a ControlMessage's metadata
    based on the presence of a 'config::add_trace_tagging' flag.

    This decorator checks if the 'config::add_trace_tagging' flag is set to True in the
    message's metadata. If so, it records the entry and exit timestamps of the function
    execution, using either a provided custom trace name or the function's name by default.

    Parameters
    ----------
    trace_name : str, optional
        A custom name for the trace entries in the message metadata. If not provided, the
        function's name is used by default.

    Returns
    -------
    decorator_trace_tagging : Callable
        A wrapper function that decorates the target function to implement trace tagging.

    Notes
    -----
    The decorated function must accept a ControlMessage object as its first argument. The
    ControlMessage object must implement `has_metadata`, `get_metadata`, and `set_metadata`
    methods used by the decorator to check for the trace tagging flag and to add trace metadata.

    The trace metadata added by the decorator includes two entries:
    - 'trace::entry::<trace_name>': The monotonic timestamp marking the function's entry.
    - 'trace::exit::<trace_name>': The monotonic timestamp marking the function's exit.

    Example
    -------
    Applying the decorator without a custom trace name:

    >>> @traceable()
    ... def process_message(message):
    ...     pass

    Applying the decorator with a custom trace name:

    >>> @traceable(custom_trace_name="CustomTraceName")
    ... def process_message(message):
    ...     pass

    In both examples, `process_message` will have entry and exit timestamps added to the
    ControlMessage's metadata if 'config::add_trace_tagging' is True.

    """

    def decorator_trace_tagging(func):
        @functools.wraps(func)
        def wrapper_trace_tagging(*args, **kwargs):
            # Assuming the first argument is always the message
            ts_fetched = datetime.now()
            message = args[0]

            do_trace_tagging = (message.has_metadata("config::add_trace_tagging") is True) and (
                message.get_metadata("config::add_trace_tagging") is True
            )

            trace_prefix = trace_name if trace_name else func.__name__

            if do_trace_tagging:
                ts_send = message.get_timestamp("latency::ts_send")
                ts_entry = datetime.now()
                message.set_timestamp(f"trace::entry::{trace_prefix}", ts_entry)
                if ts_send:
                    message.set_timestamp(f"trace::entry::{trace_prefix}_channel_in", ts_send)
                    message.set_timestamp(f"trace::exit::{trace_prefix}_channel_in", ts_fetched)

            # Call the decorated function
            result = func(*args, **kwargs)

            if do_trace_tagging:
                ts_exit = datetime.now()
                message.set_timestamp(f"trace::exit::{trace_prefix}", ts_exit)
                message.set_timestamp("latency::ts_send", ts_exit)

            return result

        return wrapper_trace_tagging

    return decorator_trace_tagging
