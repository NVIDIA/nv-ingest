# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTML ingestion: markitdown conversion to markdown, then tokenizer-based chunking.

Compatible with the same embed and LanceDB stages as PDF/txt primitives.
"""

from .convert import (
    html_bytes_to_chunks_df,
    html_file_to_chunks_df,
    html_to_markdown,
)

__all__ = [
    "html_bytes_to_chunks_df",
    "html_file_to_chunks_df",
    "html_to_markdown",
]
