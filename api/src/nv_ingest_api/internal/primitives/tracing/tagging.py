# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import functools
import inspect
import string
from datetime import datetime


def traceable(trace_name=None):
    """
    A decorator that adds entry and exit trace timestamps to a IngestControlMessage's metadata
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
    The decorated function must accept a IngestControlMessage object as one of its arguments.
    For a regular function, this is expected to be the first argument; for a class method,
    this is expected to be the second argument (after 'self'). The IngestControlMessage object
    must implement `has_metadata`, `get_metadata`, and `set_metadata` methods used by the decorator
    to check for the trace tagging flag and to add trace metadata.

    The trace metadata added by the decorator includes two entries:
    - 'trace::entry::<trace_name>': The timestamp marking the function's entry.
    - 'trace::exit::<trace_name>': The timestamp marking the function's exit.

    Example
    -------
    Applying the decorator without a custom trace name:

    >>> @traceable()
    ... def process_message(message):
    ...     pass

    Applying the decorator with a custom trace name on a class method:

    >>> class Processor:
    ...     @traceable(trace_name="CustomTrace")
    ...     def process(self, message):
    ...         pass
    """

    def decorator_trace_tagging(func):
        @functools.wraps(func)
        def wrapper_trace_tagging(*args, **kwargs):
            ts_fetched = datetime.now()
            # Determine which argument is the message.
            if hasattr(args[0], "has_metadata"):
                message = args[0]
            elif len(args) > 1 and hasattr(args[1], "has_metadata"):
                message = args[1]
            else:
                raise ValueError("traceable decorator could not find a message argument with 'has_metadata()'")

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

            # Call the decorated function.
            result = func(*args, **kwargs)

            if do_trace_tagging:
                ts_exit = datetime.now()
                message.set_timestamp(f"trace::exit::{trace_prefix}", ts_exit)
                message.set_timestamp("latency::ts_send", ts_exit)

            return result

        return wrapper_trace_tagging

    return decorator_trace_tagging


def traceable_func(trace_name=None, dedupe=True):
    """
    A decorator that injects trace information for tracking the execution of a function.
    It logs the entry and exit timestamps of the function in a `trace_info` dictionary,
    which can be used for performance monitoring or debugging purposes.

    Parameters
    ----------
    trace_name : str, optional
        An optional string used as the prefix for the trace log entries. If not provided,
        the decorated function's name is used. The string can include placeholders (e.g.,
        "pdf_extractor::{model_name}") that will be dynamically replaced with matching
        function argument values.
    dedupe : bool, optional
        If True, ensures that the trace entry and exit keys are unique by appending an index
        (e.g., `_0`, `_1`) to the keys if duplicate entries are detected. Default is True.

    Returns
    -------
    function
        A wrapped function that injects trace information before and after the function's
        execution.

    Notes
    -----
    - If `trace_info` is not provided in the keyword arguments, a new dictionary is created
      and used for storing trace entries.
    - If `trace_name` contains format placeholders, the decorator attempts to populate them
      with matching argument values from the decorated function.
    - The trace information is logged in the format:
        - `trace::entry::{trace_name}` for the entry timestamp.
        - `trace::exit::{trace_name}` for the exit timestamp.
    - If `dedupe` is True, the trace keys will be appended with an index to avoid
      overwriting existing entries.

    Example
    -------
    >>> @traceable_func(trace_name="pdf_extractor::{model_name}")
    >>> def extract_pdf(model_name):
    ...     pass
    >>> trace_info = {}
    >>> extract_pdf("my_model", trace_info=trace_info)

    In this example, `model_name` is dynamically replaced in the trace_name, and the
    trace information is logged with unique keys if deduplication is enabled.
    """

    def decorator_inject_trace_info(func):
        @functools.wraps(func)
        def wrapper_inject_trace_info(*args, **kwargs):
            trace_info = kwargs.pop("trace_info", None)
            if trace_info is None:
                trace_info = {}
            trace_prefix = trace_name if trace_name else func.__name__

            arg_names = list(inspect.signature(func).parameters)
            args_name_to_val = dict(zip(arg_names, args))

            # If `trace_name` is a formattable string, e.g., "pdf_extractor::{model_name}",
            # search `args` and `kwargs` to replace the placeholder.
            placeholders = [x[1] for x in string.Formatter().parse(trace_name) if x[1] is not None]
            if placeholders:
                format_kwargs = {}
                for name in placeholders:
                    if name in args_name_to_val:
                        arg_val = args_name_to_val[name]
                    elif name in kwargs:
                        arg_val = kwargs.get(name)
                    else:
                        arg_val = name
                    format_kwargs[name] = arg_val
                trace_prefix = trace_prefix.format(**format_kwargs)

            trace_entry_key = f"trace::entry::{trace_prefix}"
            trace_exit_key = f"trace::exit::{trace_prefix}"

            ts_entry = datetime.now()

            if dedupe:
                trace_entry_key += "_{}"
                trace_exit_key += "_{}"
                i = 0
                while (trace_entry_key.format(i) in trace_info) or (trace_exit_key.format(i) in trace_info):
                    i += 1
                trace_entry_key = trace_entry_key.format(i)
                trace_exit_key = trace_exit_key.format(i)

            trace_info[trace_entry_key] = ts_entry

            # Call the decorated function
            result = func(*args, **kwargs)

            ts_exit = datetime.now()

            trace_info[trace_exit_key] = ts_exit

            return result

        return wrapper_inject_trace_info

    return decorator_inject_trace_info
