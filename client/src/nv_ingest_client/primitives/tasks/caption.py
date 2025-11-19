# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict


from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskCaptionSchema
from .task_base import Task

logger = logging.getLogger(__name__)


class CaptionTask(Task):
    def __init__(
        self,
        api_key: str = None,
        endpoint_url: str = None,
        prompt: str = None,
        model_name: str = None,
    ) -> None:
        super().__init__()

        # Use the API schema for validation
        validated_data = IngestTaskCaptionSchema(
            api_key=api_key, endpoint_url=endpoint_url, prompt=prompt, model_name=model_name
        )

        self._api_key = validated_data.api_key
        self._endpoint_url = validated_data.endpoint_url
        self._prompt = validated_data.prompt
        self._model_name = validated_data.model_name

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Image Caption Task:\n"

        if self._api_key:
            info += "  api_key: [redacted]\n"
        if self._endpoint_url:
            info += f"  endpoint_url: {self._endpoint_url}\n"
        if self._prompt:
            info += f"  prompt: {self._prompt}\n"
        if self._model_name:
            info += f"  model_name: {self._model_name}\n"

        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        task_properties = {}

        if self._api_key:
            task_properties["api_key"] = self._api_key

        if self._endpoint_url:
            task_properties["endpoint_url"] = self._endpoint_url

        if self._prompt:
            task_properties["prompt"] = self._prompt

        if self._model_name:
            task_properties["model_name"] = self._model_name

        return {"type": "caption", "task_properties": task_properties}
