# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .ocr import (
    OCRActor,
    blocks_to_pseudo_markdown,
    blocks_to_text,
    crop_all_from_page,
    crop_b64_image_by_norm_bbox,
    extract_remote_ocr_item,
    np_rgb_to_b64_png,
    ocr_page_elements,
    parse_ocr_result,
)

__all__ = [
    "OCRActor",
    "blocks_to_pseudo_markdown",
    "blocks_to_text",
    "crop_all_from_page",
    "crop_b64_image_by_norm_bbox",
    "extract_remote_ocr_item",
    "np_rgb_to_b64_png",
    "ocr_page_elements",
    "parse_ocr_result",
]
