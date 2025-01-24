# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
import typing
from functools import wraps
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


def filter_by_task(required_tasks, forward_func=None):
    """
    A decorator that checks if the first argument to the wrapped function (expected to be a ControlMessage object)
    contains any of the tasks specified in `required_tasks`. Each task can be a string of the task name or a tuple
    of the task name and task properties. If the message does not contain any listed task and/or task properties,
    the message is returned directly without calling the wrapped function, unless a forwarding
    function is provided, in which case it calls that function on the ControlMessage.

    Parameters
    ----------
    required_tasks : list
        A list of task keys (string or tuple/list of [task_name, task_property_dict(s)]) to check for in the
        ControlMessage.
    forward_func : callable, optional
        A function to be called with the ControlMessage if no required task is found. Defaults to None.

    Returns
    -------
    callable
        The wrapped function, conditionally called based on the task check.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args or not hasattr(args[0], "get_tasks"):
                raise ValueError("The first argument must be a ControlMessage object with task handling capabilities.")

            message = args[0]
            tasks = message.get_tasks()
            logger.debug(f"Tasks in message: {list(tasks.keys())}")
            logger.debug(f"Required tasks: {required_tasks}")

            for required_task in required_tasks:
                # 1) If the required task is a string (simple check for existence)
                if isinstance(required_task, str):
                    if required_task in tasks:
                        logger.debug(f"Found required task '{required_task}'. Executing function.")
                        return func(*args, **kwargs)
                    else:
                        logger.debug(f"Task '{required_task}' not found in ControlMessage. Skipping.")

                # 2) If the required task is a tuple/list: (task_name, {prop_key: prop_val}, ...)
                elif isinstance(required_task, (tuple, list)):
                    required_task_name, *required_task_props_list = required_task
                    if required_task_name not in tasks:
                        logger.debug(f"Task '{required_task_name}' not found in ControlMessage. Skipping.")
                        continue

                    # We have at least one task of this type. Check the properties:
                    task_props_list = tasks.get(required_task_name, [])
                    logger.debug(f"Checking task properties for '{required_task_name}': {task_props_list}")
                    logger.debug(f"Required task properties: {required_task_props_list}")

                    # Check each set of task_props against the required subset(s)
                    for task_props in task_props_list:
                        if isinstance(task_props, BaseModel):
                            task_props = task_props.model_dump()

                        # We need to match *all* required_task_props in `required_task_props_list`
                        # with the current `task_props`.
                        if all(
                            _is_subset(task_props, required_task_props)
                            for required_task_props in required_task_props_list
                        ):
                            logger.debug(
                                f"Task '{required_task_name}' with properties {task_props} "
                                f"matches all required properties. Executing function."
                            )
                            return func(*args, **kwargs)
                        else:
                            logger.debug(
                                f"Task '{required_task_name}' with properties {task_props} "
                                f"does not match all required properties {required_task_props_list}. Skipping."
                            )

            # If we got here, it means none of the required tasks or properties matched
            logger.debug("No required tasks matched. Forwarding or returning message as configured.")

            if forward_func:
                # If a forward function is provided, call it with the ControlMessage
                return forward_func(message)
            else:
                # If no forward function is provided, return the message directly
                return message

        return wrapper

    return decorator


def _is_subset(superset, subset):
    if subset == "*":
        return True

    if isinstance(superset, dict) and isinstance(subset, dict):
        return all(key in superset and _is_subset(superset[key], val) for key, val in subset.items())

    if isinstance(subset, str) and subset.startswith("regex:"):
        # The subset is a regex pattern
        pattern = subset[len("regex:") :]
        if isinstance(superset, list):
            return any(re.match(pattern, str(sup_item)) for sup_item in superset)
        else:
            return re.match(pattern, str(superset)) is not None

    if isinstance(superset, list) and not isinstance(subset, list):
        # Check if the subset value matches any item in the superset
        return any(_is_subset(sup_item, subset) for sup_item in superset)

    if isinstance(superset, (list, set)) and isinstance(subset, (list, set)):
        # Check if each sub_item in `subset` is in `superset` (by subset matching)
        return all(any(_is_subset(sup_item, sub_item) for sup_item in superset) for sub_item in subset)

    return superset == subset


def remove_task_subset(ctrl_msg: typing.Any, task_type: typing.List, subset: typing.Dict):
    """
    A helper function to extract a task based on subset matching when the task might be out of order with respect to the
    Morpheus pipeline. For example, if a deduplication filter occurs before scale filtering in the pipeline, but
    the task list includes scale filtering before deduplication.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The ControlMessage object containing tasks.
    task_type : list
        The name of the ControlMessage task to operate on.
    subset : dict
        The subset of the ControlMessage task to match on.

    Returns
    -------
    dict
        A dictionary representing the matched ControlMessage task properties.
    """

    filter_tasks = []
    ctrl_msg_tasks = ctrl_msg.get_tasks()

    for task in ctrl_msg_tasks:
        if task == task_type:
            for _ in ctrl_msg_tasks[task_type]:
                task_props = ctrl_msg.remove_task(task_type)
                if _is_subset(task_props, subset):
                    logger.debug(
                        f"Removed task '{task_type}' with properties {task_props} " f"matching subset {subset}."
                    )
                    break
                filter_tasks.append(task_props)
            break

    for filter_task in filter_tasks:
        ctrl_msg.add_task(task_type, filter_task)

    return task_props
