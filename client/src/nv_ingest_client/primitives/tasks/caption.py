# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict

from pydantic import BaseModel

from .task_base import Task

logger = logging.getLogger(__name__)


class CaptionTaskSchema(BaseModel):
    class Config:
        extra = "forbid"


class CaptionTask(Task):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        task_properties = {
            "content_type": "image",
        }

        return {"type": "caption", "task_properties": task_properties}
