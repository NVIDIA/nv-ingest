# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import os

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

        embedding_nim_endpoint = os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
        if embedding_nim_endpoint == "":
            self._embedding_nim_endpoint = "https://integrate.api.nvidia.com/v1"
        elif not (embedding_nim_endpoint.startswith("http://") or embedding_nim_endpoint.startswith("https://")):
            self._embedding_nim_endpoint = "http://" + embedding_nim_endpoint + "/v1"
        else:
            self._embedding_nim_endpoint = embedding_nim_endpoint

        self._embedding_nim_model_name = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")
        self._nvidia_build_api_key = os.getenv("NVIDIA_BUILD_API_KEY", "")

        self._filter_errors = filter_errors

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Embed Task:\n"
        info += f"  embedding_nim_endpoint: {self._embedding_nim_endpoint}\n"
        info += f"  embedding_nim_model_name: {self._embedding_nim_model_name}\n"
        info += "  nvidia_build_api_key: [redacted]\n"
        info += f"  filter_errors: {self._filter_errors}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """

        task_properties = {
            "filter_errors": self._filter_errors,
        }

        if self._embedding_nim_endpoint is not None:
            task_properties["embedding_nim_endpoint"] = self._embedding_nim_endpoint
        if self._embedding_nim_model_name is not None:
            task_properties["embedding_nim_model_name"] = self._embedding_nim_model_name
        if self._nvidia_build_api_key is not None:
            task_properties["nvidia_build_api_key"] = self._nvidia_build_api_key

        return {"type": "embed", "task_properties": task_properties}
