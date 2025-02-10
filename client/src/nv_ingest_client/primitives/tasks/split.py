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


class SplitTaskSchema(BaseModel):
    tokenizer: str = "intfloat/e5-large-unsupervised"
    chunk_size: int = 1024
    chunk_overlap: int = 0

    class Config:
        extra = "forbid"


class SplitTask(Task):
    """
    Object for document splitting task
    """

    def __init__(
        self,
        tokenizer: str = "intfloat/e5-large-unsupervised",
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
    ) -> None:
        """
        Setup Split Task Config
        """
        super().__init__()
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Split Task:\n"
        info += f"  tokenizer: {self._tokenizer}\n"
        info += f"  chunk_size: {self._chunk_size}\n"
        info += f"  chunk_overlap: {self._chunk_overlap}\n"
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

        return {"type": "split", "task_properties": split_params}
