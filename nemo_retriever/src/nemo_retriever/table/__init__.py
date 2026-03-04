# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Table extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/table primitives by populating
`metadata.table_metadata.table_content` using OCR (and optionally YOLOX).
"""

from .config import (
    TableExtractionStageConfig,
    TableStructureOCRStageConfig,
    load_table_extractor_schema_from_dict,
    load_table_structure_ocr_config_from_dict,
)
from .commands import app
from .processor import extract_table_data_from_primitives_df
from .table_detection import TableStructureActor, table_structure_ocr_page_elements

__all__ = [
    "app",
    "TableExtractionStageConfig",
    "TableStructureOCRStageConfig",
    "TableStructureActor",
    "extract_table_data_from_primitives_df",
    "load_table_extractor_schema_from_dict",
    "load_table_structure_ocr_config_from_dict",
    "table_structure_ocr_page_elements",
]
