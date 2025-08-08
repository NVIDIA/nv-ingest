# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict
from typing import Literal

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskDedupSchema

from .task_base import Task

logger = logging.getLogger(__name__)


class DedupTask(Task):
    """
    Object for document dedup task
    """

    _TypeContentType = Literal["image"]

    def __init__(
        self,
        content_type: _TypeContentType = "image",
        filter: bool = False,
    ) -> None:
        """
        Setup Dedup Task Config
        """
        super().__init__()

        # Use the API schema for validation
        validated_data = IngestTaskDedupSchema(
            content_type=content_type,
            params={"filter": filter},
        )

        self._content_type = validated_data.content_type
        self._filter = validated_data.params.filter

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Dedup Task:\n"
        info += f"  content_type: {self._content_type.value}\n"
        info += f"  filter: {self._filter}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        dedup_params = {"filter": self._filter}

        task_properties = {
            "content_type": self._content_type.value,
            "params": dedup_params,
        }

        return {"type": "dedup", "task_properties": task_properties}
