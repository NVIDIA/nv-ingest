# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PDF extraction stage (pure Python + Ray Data adapters).

This package intentionally reuses the core extraction logic from `nv-ingest-api`
and only provides thin orchestration wrappers.
"""

from .__main__ import app
from .config import PDFExtractionStageConfig, load_pdf_extractor_schema_from_dict
from .io import pdf_files_to_ledger_df
from .stage import extract_pdf_primitives_from_ledger_df, make_pdf_task_config

__all__ = [
    "app",
    "PDFExtractionStageConfig",
    "extract_pdf_primitives_from_ledger_df",
    "load_pdf_extractor_schema_from_dict",
    "make_pdf_task_config",
    "pdf_files_to_ledger_df",
]
