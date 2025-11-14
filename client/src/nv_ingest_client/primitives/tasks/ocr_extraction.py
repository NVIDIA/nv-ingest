# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskOCRExtraction
from nv_ingest_client.primitives.tasks.task_base import Task

logger = logging.getLogger(__name__)


class OCRExtractionTask(Task):
    """
    Object for ocr extraction task
    """

    def __init__(self, params: dict = None) -> None:
        """
        Setup OCR Extraction Task Config
        """
        super().__init__()

        # Handle None params by converting to empty dict for backward compatibility
        if params is None:
            params = {}

        # Use the API schema for validation
        validated_data = IngestTaskOCRExtraction(params=params)

        self._params = validated_data.params

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "OCR Extraction Task:\n"
        info += f"  params: {self._params}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        task_properties = {
            "params": self._params,
        }

        return {"type": "ocr_data_extract", "task_properties": task_properties}
