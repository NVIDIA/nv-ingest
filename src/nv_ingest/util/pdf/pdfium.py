# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import traceback
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from numpy import dtype
from numpy import ndarray
from PIL import Image

from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.image_processing.transforms import pad_image
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
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


@traceable_func(trace_name="pdf_content_extractor::pdfium_pages_to_numpy")
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
        The DPI (dots per inch) at which to render the pages. Defaults to 300.
    scale_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to resize the rendered image to using the thumbnail approach.
        Defaults to None.
    padding_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to pad the image to. Defaults to None.
    rotation:

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

    for idx, page in enumerate(pages):
        # Render the page as a bitmap with the specified scale and rotation
        page_bitmap = page.render(scale=scale, rotation=rotation)

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


def convert_pdfium_position(pos, page_width, page_height):
    """
    Convert a PDFium bounding box (which typically has an origin at the bottom-left)
    to a more standard bounding-box format with y=0 at the top.
    """
    left, bottom, right, top = pos
    x0, x1 = left, right
    y0, y1 = page_height - top, page_height - bottom

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(page_width, x1)
    y1 = min(page_height, y1)

    return [int(x0), int(y0), int(x1), int(y1)]


def boxes_are_close_or_overlap(b1, b2, threshold=10.0):
    """
    Return True if bounding boxes b1 and b2 overlap or
    are closer than 'threshold' in points/pixels.
    b1, b2 = (xmin, ymin, xmax, ymax).
    """
    (xmin1, ymin1, xmax1, ymax1) = b1
    (xmin2, ymin2, xmax2, ymax2) = b2

    # Check if they overlap horizontally
    overlap_x = not (xmax1 < xmin2 or xmax2 < xmin1)
    # Check if they overlap vertically
    overlap_y = not (ymax1 < ymin2 or ymax2 < ymin1)

    # If there's an actual overlap area, that's enough
    if overlap_x and overlap_y:
        return True

    # Otherwise, measure the gap. We can do a simple approach:
    #   horizontal gap = distance between the rightmost left edge and the leftmost right edge
    #   vertical gap   = ...
    # But to keep it straightforward, let's do a quick bounding box expansions approach
    # Expand each box by 'threshold' in all directions and see if they overlap now
    expanded_b1 = (xmin1 - threshold, ymin1 - threshold, xmax1 + threshold, ymax1 + threshold)
    expanded_b2 = (xmin2 - threshold, ymin2 - threshold, xmax2 + threshold, ymax2 + threshold)

    # Check overlap on expanded boxes
    (exmin1, eymin1, exmax1, eymax1) = expanded_b1
    (exmin2, eymin2, exmax2, eymax2) = expanded_b2

    overlap_x_expanded = not (exmax1 < exmin2 or exmax2 < exmin1)
    overlap_y_expanded = not (eymax1 < eymin2 or eymax2 < eymin1)

    return overlap_x_expanded and overlap_y_expanded


def group_bounding_boxes(boxes, threshold=10.0, max_num_boxes=1_000, max_depth=None):
    """
    Given a list of bounding boxes,
    returns a list of groups (lists) of bounding box indices.
    """
    n = len(boxes)
    if n > max_num_boxes:
        logger.warning(
            "Number of bounding boxes (%d) exceeds the maximum allowed (%d). "
            "Skipping grouping to avoid high computational overhead.",
            n,
            max_num_boxes,
        )
        return []

    visited = [False] * n
    adjacency_list = [[] for _ in range(n)]

    # Build adjacency by checking closeness/overlap
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_are_close_or_overlap(boxes[i], boxes[j], threshold):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # DFS to get connected components
    def dfs(start):
        stack = [(start, 0)]  # (node, depth)
        component = []
        while stack:
            node, depth = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)

                # If we haven't reached max_depth (if max_depth is set)
                if max_depth is None or depth < max_depth:
                    for neighbor in adjacency_list[node]:
                        if not visited[neighbor]:
                            stack.append((neighbor, depth + 1))

        return component

    groups = []
    for i in range(n):
        if not visited[i]:
            comp = dfs(i)
            groups.append(comp)

    return groups


def combine_groups_into_bboxes(boxes, groups, min_num_components=1):
    """
    Given the original bounding boxes and a list of groups (each group is
    a list of indices), return one bounding box per group.
    """
    combined = []
    for group in groups:
        if len(group) < min_num_components:
            continue
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for idx in group:
            (xmin, ymin, xmax, ymax) = boxes[idx]
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        group_xmin = min(xmins)
        group_ymin = min(ymins)
        group_xmax = max(xmaxs)
        group_ymax = max(ymaxs)

        combined.append((group_xmin, group_ymin, group_xmax, group_ymax))

    return combined


def extract_simple_images_from_pdfium_page(page, max_depth):
    page_width = page.get_width()
    page_height = page.get_height()

    try:
        image_objects = page.get_objects(
            filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,),
            max_depth=max_depth,
        )
    except Exception as e:
        logger.error(f"Unhandled error extracting image: {e}")
        traceback.print_exc()
        return []

    extracted_images = []
    for obj in image_objects:
        try:
            # Attempt to retrieve the image bitmap
            image_numpy: np.ndarray = pdfium_try_get_bitmap_as_numpy(obj)  # noqa
            image_base64: str = numpy_to_base64(image_numpy)
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
            logger.error(f"Unhandled error extracting image: {e}")
            traceback.print_exc()
            pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

    return extracted_images


def extract_nested_simple_images_from_pdfium_page(page):
    return extract_simple_images_from_pdfium_page(page, max_depth=2)


def extract_top_level_simple_images_from_pdfium_page(page):
    return extract_simple_images_from_pdfium_page(page, max_depth=1)


def remove_superset_bboxes(bboxes):
    """
    Given a list of bounding boxes (x_min, y_min, x_max, y_max),
    remove any bounding box that is a strict superset of another
    (i.e., fully contains a smaller box).

    Returns:
        A new list of bounding boxes without the supersets.
    """
    results = []

    for i, box_a in enumerate(bboxes):
        xA_min, yA_min, xA_max, yA_max = box_a

        # Flag to mark if we should exclude this box
        exclude_a = False

        for j, box_b in enumerate(bboxes):
            if i == j:
                continue

            xB_min, yB_min, xB_max, yB_max = box_b

            # Check if box_a strictly encloses box_b:
            # 1) xA_min <= xB_min, yA_min <= yB_min, xA_max >= xB_max, yA_max >= yB_max
            # 2) At least one of those inequalities is strict, meaning they're not equal on all edges
            if xA_min <= xB_min and yA_min <= yB_min and xA_max >= xB_max and yA_max >= yB_max:
                # box_a is a strict superset => remove it
                exclude_a = True
                break

        if not exclude_a:
            results.append(box_a)

    return results


def extract_merged_images_from_pdfium_page(page, merge=True):
    """
    Extract bounding boxes of image objects from a PDFium page, with optional merging
    of bounding boxes that likely belong to the same compound image.
    """
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

    merged_groups = group_bounding_boxes(image_bboxes)
    merged_bboxes = combine_groups_into_bboxes(image_bboxes, merged_groups)

    return merged_bboxes


def extract_merged_shapes_from_pdfium_page(page, merge=True):
    """
    Extract bounding boxes of path objects (shapes) from a PDFium page, and optionally merge
    those bounding boxes if they appear to be part of the same shape group. Also filters out
    shapes that occupy more than half the page area.
    """
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

    path_groups = group_bounding_boxes(path_bboxes)
    path_bboxes = combine_groups_into_bboxes(path_bboxes, path_groups, min_num_components=3)
    for bbox in path_bboxes:
        bbox_area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
        # Exclude shapes that are too large (likely page backgrounds or false positives)
        if bbox_area > 0.5 * page_area:
            continue
        merged_bboxes.append(bbox)

    return merged_bboxes


def extract_forms_from_pdfium_page(page):
    """
    Extract bounding boxes for PDF form objects from a PDFium page, removing any
    bounding boxes that strictly enclose other boxes (i.e., are strict supersets).
    """
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
    form_groups = group_bounding_boxes(form_bboxes)
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


def extract_image_like_objects_from_pdfium_page(page, merge=True):
    page_width = page.get_width()
    page_height = page.get_height()
    rotation = page.get_rotation()

    try:
        original_images, _ = pdfium_pages_to_numpy(
            [page],  # A batch with a single image.
            render_dpi=72,  # dpi = 72 is equivalent to scale = 1.
            rotation=rotation,  # Without rotation, coordinates from page.get_pos() will not match.
        )
        image_bboxes = extract_merged_images_from_pdfium_page(page, merge=merge)
        shape_bboxes = extract_merged_shapes_from_pdfium_page(page, merge=merge)
        form_bboxes = extract_forms_from_pdfium_page(page)
    except Exception as e:
        logger.error(f"Unhandled error extracting image: {e}")
        traceback.print_exc()
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
            logger.error(f"Unhandled error extracting image: {e}")
            traceback.print_exc()
            pass  # Pdfium failed to extract the image associated with this object - corrupt or missing.

    return extracted_images
