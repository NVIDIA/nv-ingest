# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .audio_extraction import AudioExtractionTask
from .caption import CaptionTask
from .chart_extraction import ChartExtractionTask
from .dedup import DedupTask
from .embed import EmbedTask
from .extract import ExtractTask
from .filter import FilterTask
from .infographic_extraction import InfographicExtractionTask
from .split import SplitTask
from .store import StoreTask
from .store import StoreEmbedTask
from .table_extraction import TableExtractionTask
from .task_base import Task
from .task_base import TaskType
from .task_base import is_valid_task_type
from .task_factory import task_factory

__all__ = [
    "AudioExtractionTask",
    "CaptionTask",
    "ChartExtractionTask",
    "ExtractTask",
    "is_valid_task_type",
    "InfographicExtractionTask",
    "SplitTask",
    "StoreEmbedTask",
    "StoreTask",
    "TableExtractionTask",
    "Task",
    "task_factory",
    "TaskType",
    "DedupTask",
    "FilterTask",
    "EmbedTask",
]
