# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskSplitSchema

from .task_base import Task

logger = logging.getLogger(__name__)


class SplitTask(Task):
    """
    Object for document splitting task
    """

    def __init__(
        self,
        tokenizer: str = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 150,
        params: dict = None,
    ):
        """
        Setup Split Task Config
        """
        super().__init__()

        # Handle None params by converting to empty dict for backward compatibility
        if params is None:
            params = {}

        # Use the API schema for validation
        validated_data = IngestTaskSplitSchema(
            tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap, params=params
        )

        self._tokenizer = validated_data.tokenizer
        self._chunk_size = validated_data.chunk_size
        self._chunk_overlap = validated_data.chunk_overlap
        self._params = validated_data.params

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Split Task:\n"
        info += f"  tokenizer: {self._tokenizer}\n"
        info += f"  chunk_size: {self._chunk_size}\n"
        info += f"  chunk_overlap: {self._chunk_overlap}\n"
        for key, value in self._params.items():
            info += f"  {key}: {value}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        split_params = {}

        if self._tokenizer is not None:
            split_params["tokenizer"] = self._tokenizer
        if self._chunk_size is not None:
            split_params["chunk_size"] = self._chunk_size
        if self._chunk_overlap is not None:
            split_params["chunk_overlap"] = self._chunk_overlap
        if self._params is not None:
            split_params["params"] = self._params

        return {"type": "split", "task_properties": split_params}
