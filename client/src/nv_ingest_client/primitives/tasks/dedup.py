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
        enable_bbox_dedup: bool = True,
        iou_threshold: float = 0.45,
        prefer_structured: bool = True,
    ) -> None:
        """
        Setup Dedup Task Config

        Parameters
        ----------
        content_type : str
            Content type to deduplicate (currently only "image" supported).
        filter : bool
            Legacy filter parameter.
        enable_bbox_dedup : bool
            Enable bounding box overlap deduplication. When True, images that
            substantially overlap with structured elements (tables/charts) on
            the same page are removed.
        iou_threshold : float
            IoU (Intersection over Union) threshold for bbox dedup (0.0-1.0).
            Elements with IoU >= threshold are considered duplicates.
        prefer_structured : bool
            When True, keep tables/charts and drop overlapping images.
            When False, keep images and drop overlapping structured elements.
        """
        super().__init__()

        # Validate iou_threshold
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")

        # Use the API schema for validation
        validated_data = IngestTaskDedupSchema(
            content_type=content_type,
            params={
                "filter": filter,
                "enable_bbox_dedup": enable_bbox_dedup,
                "iou_threshold": iou_threshold,
                "bbox_dedup_prefer_structured": prefer_structured,
            },
        )

        self._content_type = validated_data.content_type
        self._filter = validated_data.params.filter
        self._enable_bbox_dedup = validated_data.params.enable_bbox_dedup
        self._iou_threshold = validated_data.params.iou_threshold
        self._prefer_structured = validated_data.params.bbox_dedup_prefer_structured

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Dedup Task:\n"
        info += f"  content_type: {self._content_type.value}\n"
        info += f"  filter: {self._filter}\n"
        info += f"  enable_bbox_dedup: {self._enable_bbox_dedup}\n"
        if self._enable_bbox_dedup:
            info += f"  iou_threshold: {self._iou_threshold}\n"
            info += f"  prefer_structured: {self._prefer_structured}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        dedup_params = {
            "filter": self._filter,
            "enable_bbox_dedup": self._enable_bbox_dedup,
            "iou_threshold": self._iou_threshold,
            "bbox_dedup_prefer_structured": self._prefer_structured,
        }

        task_properties = {
            "content_type": self._content_type.value,
            "params": dedup_params,
        }

        return {"type": "dedup", "task_properties": task_properties}
