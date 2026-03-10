# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Data adapter for images: ImageLoadActor turns bytes+path batches into page rows.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .load import image_bytes_to_pages_df

# Output columns matching the PDF extraction schema.
_PAGE_COLUMNS = [
    "path",
    "page_number",
    "source_id",
    "text",
    "page_image",
    "images",
    "tables",
    "charts",
    "infographics",
    "metadata",
]


class ImageLoadActor:
    """
    Ray Data map_batches callable: DataFrame with bytes, path -> DataFrame of page rows.

    Each output row matches the PDF extraction schema so downstream GPU stages
    (page-element detection, OCR, table/chart/infographic extraction) work unchanged.
    """

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(columns=_PAGE_COLUMNS)

        out_dfs: List[pd.DataFrame] = []
        for _, row in batch_df.iterrows():
            raw = row.get("bytes")
            path = row.get("path")
            if raw is None or path is None:
                continue
            path_str = str(path) if path is not None else ""
            try:
                page_df = image_bytes_to_pages_df(raw, path_str)
                if not page_df.empty:
                    out_dfs.append(page_df)
            except Exception:
                continue
        if not out_dfs:
            return pd.DataFrame(columns=_PAGE_COLUMNS)
        return pd.concat(out_dfs, ignore_index=True)
