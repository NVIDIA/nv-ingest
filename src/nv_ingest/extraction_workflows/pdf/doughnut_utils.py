# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from math import ceil
from math import floor
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from nv_ingest.util.image_processing.transforms import numpy_to_base64

DEFAULT_DPI = 300
DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1280

ACCEPTED_CLASSES = set(
    [
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "TOC",
        "Bibliography",
        "Formula",
    ]
)
IGNORED_CLASSES = set(
    [
        "Page-header",
        "Page-footer",
        "Caption",
        "Footnote",
        "Floating-text",
    ]
)

_re_extract_class_bbox = re.compile(
    r"<x_(\d+)><y_(\d+)>(.*?)<x_(\d+)><y_(\d+)><class_([^>]+)>", re.MULTILINE | re.DOTALL
)


def extract_classes_bboxes(text: str) -> Tuple[List[str], List[Tuple[int, int, int, int]], List[str]]:
    classes: List[str] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    texts: List[str] = []
    for m in _re_extract_class_bbox.finditer(text):
        x1, y1, text, x2, y2, cls = m.groups()
        classes.append(cls)
        bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        texts.append(text)

    return classes, bboxes, texts


def convert_mmd_to_plain_text_ours(mmd_text, remove_inline_math: bool = False):
    # Remove markdown links (e.g., [link](url))
    mmd_text = re.sub(r"\(\[https?://[^\s\]]+\]\((https?://[^\s\]]+)\)\)", r"(\1)", mmd_text)

    # Remove headers (e.g., ##)
    mmd_text = re.sub(r"#+\s", "", mmd_text)

    # Remove bold (e.g., **)
    mmd_text = mmd_text.replace("**", "")
    # Remove italic (e.g., *)
    mmd_text = re.sub(r"\*(.*?)\*", r"\1", mmd_text)
    # Remove emphasized text formatting (e.g., _)
    mmd_text = re.sub(r"(?<!\w)_([^_]+)_", r"\1", mmd_text)

    # Remove superscript and subscript
    mmd_text = re.sub(r"</?su[pb]>", "", mmd_text)

    if remove_inline_math:
        # Remove formulas inside paragraphs (e.g., \(R_{ij}(P^{a})=0\))
        mmd_text = re.sub(r"\\\((.*?)\\\)", "", mmd_text)
    else:
        # Treat simple formulas inside paragraphs as plain text
        mmd_text = re.sub(r"\\\((.*?)\\\)", r"\1", mmd_text)

    # Remove asterisk in lists
    mmd_text = re.sub(r"^\*\s", "", mmd_text, flags=re.MULTILINE)
    # Remove tables
    mmd_text = re.sub(r"\\begin{table}(.*?)\\end{table}", "", mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r"\\begin{tabular}(.*?)\\end{tabular}", "<table>", mmd_text, flags=re.DOTALL)
    # Remove code blocks (e.g., ```python ... ```)
    mmd_text = re.sub(r"```.*?```", "", mmd_text, flags=re.DOTALL)
    # Remove equations (e.g., \[ ... \])
    mmd_text = re.sub(r"\\\[(.*?)\\\]", "", mmd_text, flags=re.DOTALL)
    # Remove inline equations (e.g., $ ... $)
    mmd_text = re.sub(r"\$(.*?)\$", "", mmd_text)
    # Remove tables
    mmd_text = re.sub(r"\|.*?\|", "", mmd_text, flags=re.DOTALL)

    # Additional cleanup for special characters
    mmd_text = re.sub(r"\\", "", mmd_text)

    return mmd_text.strip()


def crop_image(array: np.array, bbox: Tuple[int, int, int, int], format="PNG") -> Optional[str]:
    w1, h1, w2, h2 = bbox
    h1 = max(floor(h1), 0)
    h2 = min(ceil(h2), array.shape[0])
    w1 = max(floor(w1), 0)
    w2 = min(ceil(w2), array.shape[1])
    if (w2 - w1 <= 0) or (h2 - h1 <= 0):
        return None
    cropped = array[h1:h2, w1:w2]
    base64_img = numpy_to_base64(cropped)

    return base64_img


def pad_image(
    array: np.array, target_width=DEFAULT_MAX_WIDTH, target_height=DEFAULT_MAX_HEIGHT
) -> Tuple[np.array, Tuple[int, int]]:
    height, width = array.shape[:2]
    if (height > target_height) or (width > target_width):
        raise ValueError(
            f"Image array is too large. Dimensions must be width <= {target_width} and height <= {target_height}."
        )

    if height == target_height and width == target_width:
        return array, (0, 0)

    pad_height = (target_height - height) // 2
    pad_width = (target_width - width) // 2
    canvas = 255 * np.ones((target_height, target_width, 3), dtype=np.uint8)
    canvas[pad_height : pad_height + height, pad_width : pad_width + width] = array  # noqa: E203

    return canvas, (pad_width, pad_height)


def reverse_transform_bbox(
    bbox: Tuple[int, int, int, int],
    bbox_offset: Tuple[int, int],
    original_width: int = DEFAULT_MAX_WIDTH,
    original_height: int = DEFAULT_MAX_HEIGHT,
) -> Tuple[int, int, int, int]:
    width_ratio = (original_width - 2 * bbox_offset[0]) / original_width
    height_ratio = (original_height - 2 * bbox_offset[1]) / original_height
    w1, h1, w2, h2 = bbox
    w1 = int((w1 - bbox_offset[0]) / width_ratio)
    h1 = int((h1 - bbox_offset[1]) / height_ratio)
    w2 = int((w2 - bbox_offset[0]) / width_ratio)
    h2 = int((h2 - bbox_offset[1]) / height_ratio)

    return (w1, h1, w2, h2)


def postprocess_text(txt: str, cls: str):
    if cls in ACCEPTED_CLASSES:
        txt = txt.replace("<tbc>", "").strip()  # remove <tbc> tokens (continued paragraphs)
        txt = convert_mmd_to_plain_text_ours(txt)
    else:
        txt = ""

    return txt
