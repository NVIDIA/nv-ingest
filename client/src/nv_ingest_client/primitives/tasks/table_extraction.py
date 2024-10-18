# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict, Optional
from typing import Literal

from pydantic import BaseModel
from pydantic import validator

from .task_base import Task

logger = logging.getLogger(__name__)


class TableExtractionSchema(BaseModel):
    content_type: Optional[str] = "image"

    @validator("content_type")
    def content_type_must_be_valid(cls, v):
        valid_criteria = ["image"]
        if v not in valid_criteria:
            raise ValueError(f"content_type must be one of {valid_criteria}")
        return v

    class Config:
        extra = "forbid"


class TableExtractionTask(Task):
    """
    Object for document dedup task
    """

    def __init__(
            self) -> None:
        """
        Setup Dedup Task Config
        """
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Table Extraction Task:\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """

        task_properties = {
            "params": {},
        }

        return {"type": "table_extract", "task_properties": task_properties}
