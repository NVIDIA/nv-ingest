# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Table extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/table primitives by populating
`metadata.table_metadata.table_content` using OCR (and optionally YOLOX).
"""

from .config import TableExtractionStageConfig, load_table_extractor_schema_from_dict
from .commands import app
from .processor import extract_table_data_from_primitives_df

__all__ = [
    "app",
    "TableExtractionStageConfig",
    "extract_table_data_from_primitives_df",
    "load_table_extractor_schema_from_dict",
]
