# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Table extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/table primitives by populating
`metadata.table_metadata.table_content` using OCR (and optionally YOLOX).
"""

from .table_detection import TableStructureActor, table_structure_ocr_page_elements

__all__ = [
    "TableStructureActor",
    "table_structure_ocr_page_elements",
]
