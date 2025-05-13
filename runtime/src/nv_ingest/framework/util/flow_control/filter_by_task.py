# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import logging
import re
from typing import Dict, List, Any, Union, Tuple, Optional, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def filter_by_task(
    required_tasks: List[Union[str, Tuple[Any, ...]]],
    forward_func: Optional[Callable[[Any], Any]] = None,
) -> Callable:
    """
    Decorator that checks whether an IngestControlMessage contains any of the required tasks.
    Supports both synchronous and asynchronous functions as well as class methods (with 'self').
    If no required task is found, the original message is returned (or forward_func is called if provided).

    Parameters
    ----------
    required_tasks : list[Union[str, Tuple[Any, ...]]]
        A list of required tasks. Each element is either a string representing a task name
        or a tuple/list where the first element is the task name and the remaining elements
        specify required task properties.
    forward_func : Optional[Callable[[Any], Any]]
        A function to be called with the IngestControlMessage if no required task is found.

    Returns
    -------
    Callable
        A decorator wrapping a function that expects an IngestControlMessage as one of its first arguments.
    """

    def decorator(func: Callable) -> Callable:
        # Helper to extract the IngestControlMessage from the arguments.
        def extract_message(args: Tuple) -> Any:
            if args and hasattr(args[0], "get_tasks"):
                return args[0]
            elif len(args) > 1 and hasattr(args[1], "get_tasks"):
                return args[1]
            else:
                raise ValueError(
                    "The first or second argument must be an IngestControlMessage with task handling capabilities."
                )

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                message = extract_message(args)
                tasks: Dict[str, List[Any]] = {}
                for task in message.get_tasks():
                    tasks.setdefault(task.type, []).append(task.properties)
                for required_task in required_tasks:
                    # Case 1: required task is a simple string.
                    if isinstance(required_task, str):
                        if required_task in tasks:
                            logger.debug(
                                "Task '%s' found. Proceeding with async function '%s'.",
                                required_task,
                                func.__name__,
                            )
                            return await func(*args, **kwargs)
                        else:
                            logger.debug("Required task '%s' not found.", required_task)
                    # Case 2: required task is a tuple/list with properties.
                    elif isinstance(required_task, (tuple, list)):
                        required_task_name, *required_task_props_list = required_task
                        if required_task_name not in tasks:
                            continue
                        for task_props in tasks.get(required_task_name, []):
                            orig_task_props = task_props
                            if isinstance(task_props, BaseModel):
                                task_props = task_props.model_dump()
                            all_match = True
                            for required_props in required_task_props_list:
                                if not _is_subset(task_props, required_props):
                                    logger.debug(
                                        "For task '%s', task properties %s do not match required subset %s.",
                                        required_task_name,
                                        orig_task_props,
                                        required_props,
                                    )
                                    all_match = False
                                    break
                            if all_match:
                                logger.debug(
                                    "Task '%s' with properties %s matched for function '%s'.",
                                    required_task_name,
                                    orig_task_props,
                                    func.__name__,
                                )
                                return await func(*args, **kwargs)
                    else:
                        logger.debug("Invalid required task type: %s", type(required_task))
                logger.debug("No required task matched for function '%s'.", func.__name__)
                if forward_func:
                    logger.debug("Calling forward function for IngestControlMessage.")
                    return await forward_func(message)
                else:
                    logger.debug("Returning original IngestControlMessage without processing.")
                    return message

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                message = extract_message(args)
                tasks: Dict[str, List[Any]] = {}
                for task in message.get_tasks():
                    tasks.setdefault(task.type, []).append(task.properties)
                for required_task in required_tasks:
                    if isinstance(required_task, str):
                        if required_task in tasks:
                            logger.debug(
                                "Task '%s' found. Proceeding with function '%s'.", required_task, func.__name__
                            )
                            return func(*args, **kwargs)
                        else:
                            logger.debug("Required task '%s' not found.", required_task)
                    elif isinstance(required_task, (tuple, list)):
                        required_task_name, *required_task_props_list = required_task
                        if required_task_name not in tasks:
                            continue
                        for task_props in tasks.get(required_task_name, []):
                            orig_task_props = task_props
                            if isinstance(task_props, BaseModel):
                                task_props = task_props.model_dump()
                            all_match = True
                            for required_props in required_task_props_list:
                                if not _is_subset(task_props, required_props):
                                    logger.debug(
                                        "For task '%s', task properties %s do not match required subset %s.",
                                        required_task_name,
                                        orig_task_props,
                                        required_props,
                                    )
                                    all_match = False
                                    break
                            if all_match:
                                logger.debug(
                                    "Task '%s' with properties %s matched for function '%s'.",
                                    required_task_name,
                                    orig_task_props,
                                    func.__name__,
                                )
                                return func(*args, **kwargs)
                    else:
                        logger.debug("Invalid required task type: %s", type(required_task))
                logger.debug("No required task matched for function '%s'.", func.__name__)
                if forward_func:
                    logger.debug("Calling forward function for IngestControlMessage.")
                    return forward_func(message)
                else:
                    logger.debug("Returning original IngestControlMessage without processing.")
                    return message

            return sync_wrapper

    return decorator


def _is_subset(superset: Any, subset: Any) -> bool:
    """
    Recursively checks whether 'subset' is contained within 'superset'. Supports dictionaries,
    lists, strings (including regex patterns), and basic types.

    Parameters
    ----------
    superset : Any
        The data structure (or value) that is expected to contain the subset.
    subset : Any
        The data structure (or value) to be checked for being a subset of 'superset'. A special
        value "*" matches any value, and strings prefixed with "regex:" are treated as regular
        expression patterns.

    Returns
    -------
    bool
        True if 'subset' is contained within 'superset', False otherwise.
    """
    if subset == "*":
        return True
    if isinstance(superset, dict) and isinstance(subset, dict):
        for key, val in subset.items():
            if key not in superset:
                logger.debug("Key '%s' not found in superset dictionary: %s", key, superset)
                return False
            if not _is_subset(superset[key], val):
                logger.debug("Value for key '%s' (%s) does not match expected subset (%s).", key, superset[key], val)
                return False
        return True
    if isinstance(subset, str) and subset.startswith("regex:"):
        pattern = subset[len("regex:") :]
        if isinstance(superset, list):
            for sup_item in superset:
                if re.match(pattern, sup_item):
                    return True
            logger.debug("No items in list %s match regex pattern '%s'.", superset, pattern)
            return False
        else:
            if re.match(pattern, superset) is None:
                logger.debug("Value '%s' does not match regex pattern '%s'.", superset, pattern)
                return False
            return True
    if isinstance(superset, list) and not isinstance(subset, list):
        for sup_item in superset:
            if _is_subset(sup_item, subset):
                return True
        logger.debug("None of the items in list %s match the value '%s'.", superset, subset)
        return False
    if isinstance(superset, (list, set)) and isinstance(subset, list):
        for sub_item in subset:
            if not any(_is_subset(sup_item, sub_item) for sup_item in superset):
                logger.debug("No element in %s matches subset element '%s'.", superset, sub_item)
                return False
        return True
    if superset != subset:
        logger.debug("Direct comparison failed: %s != %s", superset, subset)
    return superset == subset
