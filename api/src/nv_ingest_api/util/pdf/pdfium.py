# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Any
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from numpy import dtype
from numpy import ndarray

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.util.image_processing.clustering import (
    group_bounding_boxes,
    combine_groups_into_bboxes,
    remove_superset_bboxes,
)
from nv_ingest_api.util.image_processing.transforms import pad_image, numpy_to_base64, crop_image, scale_numpy_image
from nv_ingest_api.util.metadata.aggregators import Base64Image
from nv_ingest_api.internal.primitives.nim.model_interface.yolox import YOLOX_PAGE_IMAGE_FORMAT

logger = logging.getLogger(__name__)

PDFIUM_PAGEOBJ_MAPPING = {
    pdfium_c.FPDF_PAGEOBJ_TEXT: "TEXT",
    pdfium_c.FPDF_PAGEOBJ_PATH: "PATH",
    pdfium_c.FPDF_PAGEOBJ_IMAGE: "IMAGE",
    pdfium_c.FPDF_PAGEOBJ_SHADING: "SHADING",
    pdfium_c.FPDF_PAGEOBJ_FORM: "FORM",
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
    mode = bitmap.mode  # Use the mode to identify the correct format

    # Convert to a NumPy array using the built-in method
    img_arr = bitmap.to_numpy().copy()

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
        # logger.debug("Attempting to get rendered bitmap.")
        image_bitmap = image_obj.get_bitmap(render=True)
    except pdfium.PdfiumError as e:
        logger.debug(f"Failed to get rendered bitmap: {e}")

    # If rendering failed or returned None, try without rendering
    if image_bitmap is None:
        try:
            # logger.debug("Attempting to get raw bitmap without rendering.")
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


@traceable_func(trace_name="pdf_extraction::pdfium_pages_to_numpy")
def pdfium_pages_to_numpy(
    pages: List[pdfium.PdfPage],
    render_dpi: int = 300,
    scale_tuple: Optional[Tuple[int, int]] = None,
    padding_tuple: Optional[Tuple[int, int]] = None,
    rotation: int = 0,
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
        The DPI (dots per inch) at which to render the pages. Must be between 50 and 1200.
        Defaults to 300.
    scale_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to resize the rendered image to using the thumbnail approach.
        Defaults to None.
    padding_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to pad the image to. Defaults to None.
    rotation:

    Returns
    -------
    tuple
        A tuple containing:
            - A list of NumPy arrays, where each array corresponds to an image of a PDF page.
              Each array is an independent copy of the rendered image data.
            - A list of padding offsets applied to each image, as tuples of (offset_width, offset_height).

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

    for idx, page in enumerate(pages):
        # Render the page as a bitmap with the specified scale and rotation
        page_bitmap = page.render(scale=scale, rotation=rotation)
        img_arr = convert_bitmap_to_corrected_numpy(page_bitmap)
        # Apply scaling using the thumbnail approach if specified
        if scale_tuple:
            img_arr = scale_numpy_image(img_arr, scale_tuple)
        # Apply padding if specified
        if padding_tuple:
            img_arr, (pad_width, pad_height) = pad_image(
                img_arr, target_width=padding_tuple[0], target_height=padding_tuple[1]
            )
            padding_offsets.append((pad_width, pad_height))
        else:
            padding_offsets.append((0, 0))

        images.append(img_arr)

    return images, padding_offsets


def convert_pdfium_position(pos, page_width, page_height):
    """
    Convert a PDFium bounding box (which typically has an origin at the bottom-left)
    to a more standard bounding-box format with y=0 at the top.

    Note:
        This method assumes the PDF coordinate system follows the common convention
        where the origin is at the bottom-left. However, per the PDF specification,
        the coordinate system can theoretically be defined between any opposite corners,
        and its origin may not necessarily be (0,0). This implementation may not handle
        all edge cases where the coordinate system is arbitrarily transformed.

        Further processing may be necessary downstream, particularly in filtering or
        deduplication stages, to account for variations in coordinate transformations
        and ensure consistent bounding-box comparisons.

        See https://github.com/pypdfium2-team/pypdfium2/discussions/284.
    """
    left, bottom, right, top = pos
    x0, x1 = left, right
    y0, y1 = page_height - top, page_height - bottom

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_width, x1)
    y1 = min(page_height, y1)

    return [int(x0), int(y0), int(x1), int(y1)]


def extract_simple_images_from_pdfium_page(page, max_depth):
    page_width = page.get_width()
    page_height = page.get_height()

    try:
        image_objects = page.get_objects(
            filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,),
            max_depth=max_depth,
        )
    except Exception as e:
        logger.exception(f"Unhandled error extracting image: {e}")
        return []

    extracted_images = []
    for obj in image_objects:
        try:
            # Attempt to retrieve the image bitmap
            image_numpy: np.ndarray = pdfium_try_get_bitmap_as_numpy(obj)  # noqa
            image_base64: str = numpy_to_base64(image_numpy, format=YOLOX_PAGE_IMAGE_FORMAT)
            image_bbox = obj.get_pos()
            image_size = obj.get_size()
            if image_size[0] < 10 and image_size[1] < 10:
                continue

            image_data = Base64Image(
                image=image_base64,
                bbox=image_bbox,
                width=image_size[0],
                height=image_size[1],
                max_width=page_width,
                max_height=page_height,
            )
            extracted_images.append(image_data)
        except Exception as e:
            logger.exception(f"Unhandled error extracting image: {e}")
            pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

    return extracted_images


def extract_nested_simple_images_from_pdfium_page(page):
    return extract_simple_images_from_pdfium_page(page, max_depth=2)


def extract_top_level_simple_images_from_pdfium_page(page):
    return extract_simple_images_from_pdfium_page(page, max_depth=1)


def extract_merged_images_from_pdfium_page(page, merge=True, **kwargs):
    """
    Extract bounding boxes of image objects from a PDFium page, with optional merging
    of bounding boxes that likely belong to the same compound image.
    """
    threshold = kwargs.get("images_threshold", 10.0)
    max_num_boxes = kwargs.get("images_max_num_boxes", 1_024)

    page_width = page.get_width()
    page_height = page.get_height()

    image_bboxes = []
    for obj in page.get_objects(
        filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,),
        max_depth=1,
    ):
        image_bbox = convert_pdfium_position(obj.get_pos(), page_width, page_height)
        image_bboxes.append(image_bbox)

    # If no merging is requested or no bounding boxes exist, return the list as is
    if (not merge) or (not image_bboxes):
        return image_bboxes

    merged_groups = group_bounding_boxes(image_bboxes, threshold=threshold, max_num_boxes=max_num_boxes)
    merged_bboxes = combine_groups_into_bboxes(image_bboxes, merged_groups)

    return merged_bboxes


def extract_merged_shapes_from_pdfium_page(page, merge=True, **kwargs):
    """
    Extract bounding boxes of path objects (shapes) from a PDFium page, and optionally merge
    those bounding boxes if they appear to be part of the same shape group. Also filters out
    shapes that occupy more than half the page area.
    """
    threshold = kwargs.get("shapes_threshold", 10.0)
    max_num_boxes = kwargs.get("shapes_max_num_boxes", 2_048)
    min_num_components = kwargs.get("shapes_min_num_components", 3)

    page_width = page.get_width()
    page_height = page.get_height()
    page_area = page_width * page_height

    path_bboxes = []
    for obj in page.get_objects(
        filter=(pdfium_c.FPDF_PAGEOBJ_PATH,),
        max_depth=1,
    ):
        path_bbox = convert_pdfium_position(obj.get_pos(), page_width, page_height)
        path_bboxes.append(path_bbox)

    # If merging is disabled or no bounding boxes were found, return them as-is
    if (not merge) or (not path_bboxes):
        return path_bboxes

    merged_bboxes = []

    path_groups = group_bounding_boxes(path_bboxes, threshold=threshold, max_num_boxes=max_num_boxes)
    path_bboxes = combine_groups_into_bboxes(path_bboxes, path_groups, min_num_components=min_num_components)
    for bbox in path_bboxes:
        bbox_area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
        # Exclude shapes that are too large (likely page backgrounds or false positives)
        if bbox_area > 0.5 * page_area:
            continue
        merged_bboxes.append(bbox)

    return merged_bboxes


def extract_forms_from_pdfium_page(page, **kwargs):
    """
    Extract bounding boxes for PDF form objects from a PDFium page, removing any
    bounding boxes that strictly enclose other boxes (i.e., are strict supersets).
    """
    threshold = kwargs.get("forms_threshold", 10.0)
    max_num_boxes = kwargs.get("forms_max_num_boxes", 1_024)

    page_width = page.get_width()
    page_height = page.get_height()
    page_area = page_width * page_height

    form_bboxes = []
    for obj in page.get_objects(
        filter=(pdfium_c.FPDF_PAGEOBJ_FORM,),
        max_depth=1,
    ):
        form_bbox = convert_pdfium_position(obj.get_pos(), page_width, page_height)
        form_bboxes.append(form_bbox)

    merged_bboxes = []
    form_groups = group_bounding_boxes(form_bboxes, threshold=threshold, max_num_boxes=max_num_boxes)
    form_bboxes = combine_groups_into_bboxes(form_bboxes, form_groups)
    for bbox in form_bboxes:
        bbox_area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
        # Exclude shapes that are too large (likely page backgrounds or false positives)
        if bbox_area > 0.5 * page_area:
            continue
        merged_bboxes.append(bbox)

    # Remove any bounding box that strictly encloses another.
    # The larger one is likely a background.
    results = remove_superset_bboxes(merged_bboxes)

    return results


def extract_image_like_objects_from_pdfium_page(page, merge=True, **kwargs):
    page_width = page.get_width()
    page_height = page.get_height()
    rotation = page.get_rotation()

    try:
        original_images, _ = pdfium_pages_to_numpy(
            [page],  # A batch with a single image.
            render_dpi=72,  # dpi = 72 is equivalent to scale = 1.
            rotation=rotation,  # Without rotation, coordinates from page.get_pos() will not match.
        )
        image_bboxes = extract_merged_images_from_pdfium_page(page, merge=merge, **kwargs)
        shape_bboxes = extract_merged_shapes_from_pdfium_page(page, merge=merge, **kwargs)
        form_bboxes = extract_forms_from_pdfium_page(page, **kwargs)
    except Exception as e:
        logger.exception(f"Unhandled error extracting image: {e}")
        return []

    extracted_images = []
    for bbox in image_bboxes + shape_bboxes + form_bboxes:
        try:
            cropped_image = crop_image(original_images[0], bbox, min_width=10, min_height=10)
            if cropped_image is None:  # Small images are filtered out.
                continue
            image_base64 = numpy_to_base64(cropped_image)
            image_data = Base64Image(
                image=image_base64,
                bbox=bbox,
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                max_width=page_width,
                max_height=page_height,
            )
            extracted_images.append(image_data)
        except Exception as e:
            logger.exception(f"Unhandled error extracting image: {e}")
            pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

    return extracted_images


def is_scanned_page(page) -> bool:
    tp = page.get_textpage()
    text = tp.get_text_bounded() or ""
    num_chars = len(text.strip())
    num_images = sum(1 for obj in page.get_objects() if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE)

    return num_chars == 0 and num_images > 0
