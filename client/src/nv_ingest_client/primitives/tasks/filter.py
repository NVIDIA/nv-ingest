# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict
from typing import Literal
from typing import Union

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskFilterSchema

from .task_base import Task

logger = logging.getLogger(__name__)


class FilterTask(Task):
    """
    Object for document filter task
    """

    _TypeContentType = Literal["image"]

    def __init__(
        self,
        content_type: _TypeContentType = "image",
        min_size: int = 128,
        max_aspect_ratio: Union[int, float] = 5.0,
        min_aspect_ratio: Union[int, float] = 0.2,
        filter: bool = True,
    ) -> None:
        """
        Setup Filter Task Config
        """
        super().__init__()

        # Use the API schema for validation
        validated_data = IngestTaskFilterSchema(
            content_type=content_type,
            params={
                "min_size": min_size,
                "max_aspect_ratio": max_aspect_ratio,
                "min_aspect_ratio": min_aspect_ratio,
                "filter": filter,
            },
        )

        self._content_type = validated_data.content_type
        self._min_size = validated_data.params.min_size
        self._max_aspect_ratio = validated_data.params.max_aspect_ratio
        self._min_aspect_ratio = validated_data.params.min_aspect_ratio
        self._filter = validated_data.params.filter

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Filter Task:\n"
        info += f"  content_type: {self._content_type.value}\n"
        info += f"  min_size: {self._min_size}\n"
        info += f"  max_aspect_ratio: {self._max_aspect_ratio}\n"
        info += f"  min_aspect_ratio: {self._min_aspect_ratio}\n"
        info += f"  filter: {self._filter}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        filter_params = {
            "min_size": self._min_size,
            "max_aspect_ratio": self._max_aspect_ratio,
            "min_aspect_ratio": self._min_aspect_ratio,
            "filter": self._filter,
        }

        task_properties = {
            "content_type": self._content_type.value,
            "params": filter_params,
        }

        return {"type": "filter", "task_properties": task_properties}
