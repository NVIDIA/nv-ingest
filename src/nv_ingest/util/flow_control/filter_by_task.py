# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import re
from typing import Dict, List, Any, Union, Tuple, Optional, Callable
from functools import wraps

from pydantic import BaseModel
from morpheus.messages import ControlMessage

logger = logging.getLogger(__name__)


def filter_by_task(
    required_tasks: List[Union[str, Tuple[Any, ...]]],
    forward_func: Optional[Callable[[ControlMessage], ControlMessage]] = None,
) -> Callable:
    """
    Decorator that checks whether the first argument (a ControlMessage) contains any of the
    required tasks. Each required task can be specified as a string (the task name) or as a tuple/list
    with the task name as the first element and additional task properties as subsequent elements.
    If the ControlMessage does not match any required task (and its properties), the wrapped function
    is not called; instead, the original message is returned (or a forward function is invoked, if provided).

    Parameters
    ----------
    required_tasks : list[Union[str, Tuple[Any, ...]]]
        A list of required tasks. Each element is either a string representing a task name or a tuple/list
        where the first element is the task name and the remaining elements specify required task properties.
    forward_func : Optional[Callable[[ControlMessage], ControlMessage]], optional
        A function to be called with the ControlMessage if no required task is found. Defaults to None.

    Returns
    -------
    Callable
        A decorator that wraps a function expecting a ControlMessage as its first argument.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ControlMessage:
            if args and hasattr(args[0], "get_tasks"):
                message: ControlMessage = args[0]
                tasks: Dict[str, Any] = message.get_tasks()
                for required_task in required_tasks:
                    # Case 1: required task is a simple string.
                    if isinstance(required_task, str):
                        if required_task in tasks:
                            logger.debug(
                                "Task '%s' found in ControlMessage tasks. Proceeding with function '%s'.",
                                required_task,
                                func.__name__,
                            )
                            return func(*args, **kwargs)
                        else:
                            logger.debug(
                                "Required task '%s' not found in ControlMessage tasks: %s",
                                required_task,
                                list(tasks.keys()),
                            )
                    # Case 2: required task is a tuple/list with properties.
                    elif isinstance(required_task, (tuple, list)):
                        required_task_name, *required_task_props_list = required_task
                        if required_task_name not in tasks:
                            logger.debug(
                                "Required task '%s' not present among ControlMessage tasks: %s",
                                required_task_name,
                                list(tasks.keys()),
                            )
                            continue

                        task_props_list = tasks.get(required_task_name, [])
                        logger.debug(
                            "Checking task properties for task '%s'. Found properties: %s; required: %s",
                            required_task_name,
                            task_props_list,
                            required_task_props_list,
                        )
                        for task_props in task_props_list:
                            orig_task_props = task_props
                            if BaseModel is not None and isinstance(task_props, BaseModel):
                                task_props = task_props.model_dump()
                            # Check if every required property is a subset of the task properties.
                            all_match = True
                            for required_task_props in required_task_props_list:
                                if not _is_subset(task_props, required_task_props):
                                    logger.debug(
                                        "For task '%s', task properties %s do not match required subset %s.",
                                        required_task_name,
                                        orig_task_props,
                                        required_task_props,
                                    )
                                    all_match = False
                                    break
                            if all_match:
                                logger.debug(
                                    "Task '%s' with properties %s matched the required filter for function '%s'.",
                                    required_task_name,
                                    orig_task_props,
                                    func.__name__,
                                )
                                return func(*args, **kwargs)
                    else:
                        logger.debug(
                            "Invalid type for required task filter: %s (expected str, tuple, or list).",
                            type(required_task),
                        )
                # No required task was matched.
                logger.debug("No required task matched for function '%s'.", func.__name__)
                if forward_func:
                    logger.debug("Calling forward function for ControlMessage.")
                    return forward_func(message)
                else:
                    logger.debug("Returning original ControlMessage without processing.")
                    return message
            else:
                raise ValueError("The first argument must be a ControlMessage object with task handling capabilities.")

        return wrapper

    return decorator


def _is_subset(superset: Any, subset: Any) -> bool:
    """
    Recursively checks whether 'subset' is contained within 'superset'. Supports dictionaries,
    lists, strings (including regex patterns), and basic types.

    Debug messages are printed to indicate mismatches between the superset and subset.

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
        # Check if every element in the subset list is contained in the superset.
        for sub_item in subset:
            if not any(_is_subset(sup_item, sub_item) for sup_item in superset):
                logger.debug("No element in %s matches subset element '%s'.", superset, sub_item)
                return False
        return True
    if superset != subset:
        logger.debug("Direct comparison failed: %s != %s", superset, subset)
    return superset == subset


def remove_task_subset(
    ctrl_msg: ControlMessage,
    task_type: str,
    subset: Dict[Any, Any],
) -> Dict[Any, Any]:
    """
    Removes and returns the first task (of the specified task type) from the given ControlMessage whose properties
    match the provided subset. If tasks are removed that do not match, they are re-added to the ControlMessage.
    This function is useful when the order of tasks in the ControlMessage may not align with the processing pipeline.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The ControlMessage object containing tasks.
    task_type : str
        The name of the ControlMessage task to operate on.
    subset : dict
        A dictionary representing the subset of task properties to match.

    Returns
    -------
    dict
        The dictionary of task properties for the matched task. If no task matches, the behavior is undefined
        (depending on the underlying implementation of `ctrl_msg.remove_task`).

    Notes
    -----
    This function iterates over tasks in the ControlMessage corresponding to `task_type` and removes tasks one-by-one.
    If a removed task does not match the provided subset, it is re-added to the ControlMessage.
    """
    filter_tasks: List[Any] = []
    ctrl_msg_tasks: Dict[Any, Any] = ctrl_msg.get_tasks()
    task_props: Dict[Any, Any] = {}

    for task in ctrl_msg_tasks:
        if task == task_type:
            for _ in ctrl_msg_tasks[task_type]:
                task_props = ctrl_msg.remove_task(task_type)
                if _is_subset(task_props, subset):
                    logger.debug(
                        "Task of type '%s' with properties %s matches subset %s; removing it from the ControlMessage.",
                        task_type,
                        task_props,
                        subset,
                    )
                    break
                else:
                    logger.debug(
                        "Removed task properties %s do not match subset %s; will re-add later.",
                        task_props,
                        subset,
                    )
                    filter_tasks.append(task_props)
            break

    for filter_task in filter_tasks:
        ctrl_msg.add_task(task_type, filter_task)
        logger.debug(
            "Re-added task of type '%s' with properties %s back to the ControlMessage.",
            task_type,
            filter_task,
        )

    return task_props
