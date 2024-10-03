# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as pdfium
from numpy import dtype
from numpy import ndarray
from PIL import Image

from nv_ingest.util.image_processing.transforms import pad_image
from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)

# Mapping based on the FPDF_PAGEOBJ_* constants
PDFIUM_PAGEOBJ_MAPPING = {
    1: "TEXT",  # FPDF_PAGEOBJ_TEXT
    2: "PATH",  # FPDF_PAGEOBJ_PATH
    3: "IMAGE",  # FPDF_PAGEOBJ_IMAGE
    4: "SHADING",  # FPDF_PAGEOBJ_SHADING
    5: "FORM",  # FPDF_PAGEOBJ_FORM
}


def convert_bitmap_to_corrected_numpy(bitmap: pdfium.PdfBitmap) -> np.ndarray:
    """
    Converts a PdfBitmap to a correctly formatted NumPy array, handling any necessary
    channel swapping based on the bitmap's mode.

    Parameters
    ----------
    bitmap : pdfium.PdfBitmap
        The bitmap object rendered from a PDF page.

    Returns
    -------
    np.ndarray
        A NumPy array representing the correctly formatted image data.
    """
    # Get the bitmap format information
    bitmap_info = bitmap.get_info()
    mode = bitmap_info.mode  # Use the mode to identify the correct format

    # Convert to a NumPy array using the built-in method
    img_arr = bitmap.to_numpy()

    # Automatically handle channel swapping if necessary
    if mode in {"BGRA", "BGRX"}:
        img_arr = img_arr[..., [2, 1, 0, 3]]  # Swap BGR(A) to RGB(A)
    elif mode == "BGR":
        img_arr = img_arr[..., [2, 1, 0]]  # Swap BGR to RGB

    return img_arr


def pdfium_try_get_bitmap_as_numpy(image_obj) -> np.ndarray:
    """
    Attempts to retrieve the bitmap from a PdfImage object and convert it to a NumPy array,
    first with rendering enabled and then without rendering if the first attempt fails.

    Parameters
    ----------
    image_obj : PdfImage
        The PdfImage object from which to extract the bitmap.

    Returns
    -------
    np.ndarray
        The extracted bitmap as a NumPy array.

    Raises
    ------
    PdfiumError
        If an exception occurs during bitmap retrieval and both attempts fail.

    Notes
    -----
    This function first tries to retrieve the bitmap with rendering enabled (`render=True`).
    If that fails or the bitmap returned is `None`, it attempts to retrieve the raw bitmap
    without rendering (`render=False`).
    Any errors encountered during these attempts are logged at the debug level.
    """
    image_bitmap = None

    # First attempt with rendering enabled
    try:
        logger.debug("Attempting to get rendered bitmap.")
        image_bitmap = image_obj.get_bitmap(render=True)
    except pdfium.PdfiumError as e:
        logger.debug(f"Failed to get rendered bitmap: {e}")

    # If rendering failed or returned None, try without rendering
    if image_bitmap is None:
        try:
            logger.debug("Attempting to get raw bitmap without rendering.")
            image_bitmap = image_obj.get_bitmap(render=False)
        except pdfium.PdfiumError as e:
            logger.debug(f"Failed to get raw bitmap: {e}")
            raise  # Re-raise the exception to ensure the failure is handled upstream

    # Final check if bitmap is still None
    if image_bitmap is None:
        logger.debug("Failed to obtain bitmap from the image object after both attempts.")
        raise ValueError("Failed to retrieve bitmap from the PdfImage object.")

    # Convert the bitmap to a NumPy array
    img_array = convert_bitmap_to_corrected_numpy(image_bitmap)

    return img_array


@traceable_func(trace_name="pdf_content_extractor::pdfium_pages_to_numpy")
def pdfium_pages_to_numpy(
    pages: List[pdfium.PdfPage],
    render_dpi=300,
    scale_tuple: Optional[Tuple[int, int]] = None,
    padding_tuple: Optional[Tuple[int, int]] = None,
) -> tuple[list[ndarray | ndarray[Any, dtype[Any]]], list[tuple[int, int]]]:
    """
    Converts a list of PdfPage objects to a list of NumPy arrays, where each array
    represents an image of the corresponding PDF page.

    The function renders each page as a bitmap, converts it to a PIL image, applies any
    specified scaling using the thumbnail approach, and adds padding if requested. The
    DPI for rendering can be specified, with a default value of 300 DPI.

    Parameters
    ----------
    pages : List[pdfium.PdfPage]
        A list of PdfPage objects to be rendered and converted into NumPy arrays.
    render_dpi : int, optional
        The DPI (dots per inch) at which to render the pages. Defaults to 300.
    scale_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to resize the rendered image to using the thumbnail approach.
        Defaults to None.
    padding_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to pad the image to. Defaults to None.

    Returns
    -------
    List[np.ndarray]
        A list of NumPy arrays, where each array corresponds to an image of a PDF page.

    Raises
    ------
    ValueError
        If the render_dpi is outside the allowed range (50-1200).
    PdfiumError
        If there is an issue rendering the page or converting it to a NumPy array.
    IOError
        If there is an error saving the image to disk.
    """
    if not (50 <= render_dpi <= 1200):
        raise ValueError("render_dpi must be between 50 and 1200.")

    images = []
    padding_offsets = []
    scale = render_dpi / 72  # 72 DPI is the base DPI in PDFium

    for page in pages:
        # Render the page as a bitmap with the specified scale
        page_bitmap = page.render(scale=scale, rotation=0)

        # Convert the bitmap to a PIL image
        pil_image = page_bitmap.to_pil()

        # Apply scaling using the thumbnail approach if specified
        if scale_tuple:
            pil_image.thumbnail(scale_tuple, Image.LANCZOS)

        # Convert the PIL image to a NumPy array
        img_arr = np.array(pil_image)

        # Apply padding if specified
        if padding_tuple:
            img_arr, padding_offset = pad_image(img_arr, target_width=padding_tuple[0], target_height=padding_tuple[1])
            padding_offsets.append(padding_offset)
        else:
            padding_offsets.append((0, 0))

        images.append(img_arr)

    return images, padding_offsets
