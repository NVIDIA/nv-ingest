# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict

from pydantic import BaseModel, root_validator

from .task_base import Task

logger = logging.getLogger(__name__)


class EmbedTaskSchema(BaseModel):
    filter_errors: bool = False

    @root_validator(pre=True)
    def handle_deprecated_fields(cls, values):
        if "text" in values:
            logger.warning(
                "'text' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )
            values.pop("text")
        if "tables" in values:
            logger.warning(
                "'tables' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )
            values.pop("tables")
        return values

    class Config:
        extra = "forbid"


class EmbedTask(Task):
    """
    Object for document embedding task
    """

    def __init__(self, text: bool = None, tables: bool = None, filter_errors: bool = False) -> None:
        """
        Setup Embed Task Config
        """
        super().__init__()

        if text is not None:
            logger.warning(
                "'text' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )

        if tables is not None:
            logger.warning(
                "'tables' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )

        self._filter_errors = filter_errors

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Embed Task:\n"
        info += f"  filter_errors: {self._filter_errors}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """

        task_properties = {
            "filter_errors": False,
        }

        return {"type": "embed", "task_properties": task_properties}
