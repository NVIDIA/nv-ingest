# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from enum import Enum
from enum import auto
from typing import Dict

logger = logging.getLogger(__name__)


class TaskType(Enum):
    CAPTION = auto()
    CHART_DATA_EXTRACT = auto()
    DEDUP = auto()
    EMBED = auto()
    EXTRACT = auto()
    FILTER = auto()
    INFOGRAPHIC_DATA_EXTRACT = auto()
    SPLIT = auto()
    STORE = auto()
    STORE_EMBEDDING = auto()
    TABLE_DATA_EXTRACT = auto()
    TRANSFORM = auto()
    UDF = auto()
    VDB_UPLOAD = auto()


def is_valid_task_type(task_type_str: str) -> bool:
    """
    Checks if the provided string is a valid TaskType enum value.

    Parameters
    ----------
    task_type_str : str
        The string to check against the TaskType enum values.

    Returns
    -------
    bool
        True if the string is a valid TaskType enum value, False otherwise.
    """
    return task_type_str in TaskType.__members__


class Task:
    """
    Generic task Object
    """

    def __init__(self) -> None:
        """
        Setup Ingest Task Config
        """

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += f"{self.__class__.__name__}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Returns a string with the task specification. This string is used for constructing
        tasks that are then submitted to the redis client
        """
        return {}
