# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .docx_extractor import DocxExtractorLoaderFactory
from .pdf_extractor import PDFExtractorLoaderFactory

__all__ = [
    "PDFExtractorLoaderFactory",
    "DocxExtractorLoaderFactory",
]
