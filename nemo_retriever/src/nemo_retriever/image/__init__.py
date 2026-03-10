# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Image ingestion: standalone image files (PNG, JPEG, BMP, TIFF, SVG) to page DataFrame.

Converts each image into the same single-row DataFrame schema that
``pdf_extraction()`` produces, so all downstream GPU stages (page-element
detection, OCR, table/chart/infographic extraction, embedding) work unchanged.
"""

from .load import (
    image_bytes_to_pages_df,
    image_file_to_pages_df,
)

__all__ = [
    "image_bytes_to_pages_df",
    "image_file_to_pages_df",
]
