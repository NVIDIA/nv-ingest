# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from math import ceil
from math import floor
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image

from nv_ingest.util.converters import bytetools

DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1280


def pad_image(
    array: np.ndarray, target_width: int = DEFAULT_MAX_WIDTH, target_height: int = DEFAULT_MAX_HEIGHT
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pads a NumPy array representing an image to the specified target dimensions.

    If the target dimensions are smaller than the image dimensions, no padding will be applied
    in that dimension. If the target dimensions are larger, the image will be centered within the
    canvas of the specified target size, with the remaining space filled with white padding.

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array of shape (H, W, C).
    target_width : int, optional
        The desired target width of the padded image. Defaults to DEFAULT_MAX_WIDTH.
    target_height : int, optional
        The desired target height of the padded image. Defaults to DEFAULT_MAX_HEIGHT.

    Returns
    -------
    padded_array : np.ndarray
        The padded image as a NumPy array of shape (target_height, target_width, C).
    padding_offsets : Tuple[int, int]
        A tuple containing the horizontal and vertical offsets (pad_width, pad_height) applied to center the image.

    Notes
    -----
    If the target dimensions are smaller than the current image dimensions, no padding will be applied
    in that dimension, and the image will retain its original size in that dimension.

    Examples
    --------
    >>> image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    >>> padded_image, offsets = pad_image(image, target_width=1000, target_height=1000)
    >>> padded_image.shape
    (1000, 1000, 3)
    >>> offsets
    (100, 200)
    """
    height, width = array.shape[:2]

    # Determine the padding needed, if any, while ensuring no padding is applied if the target is smaller
    pad_height = max((target_height - height) // 2, 0)
    pad_width = max((target_width - width) // 2, 0)

    # Determine final canvas size (may be equal to original if target is smaller)
    final_height = max(height, target_height)
    final_width = max(width, target_width)

    # Create the canvas and place the original image on it
    canvas = 255 * np.ones((final_height, final_width, array.shape[2]), dtype=np.uint8)
    canvas[pad_height : pad_height + height, pad_width : pad_width + width] = array  # noqa: E203

    return canvas, (pad_width, pad_height)


def crop_image(
    array: np.array, bbox: Tuple[int, int, int, int], min_width: int = 1, min_height: int = 1
) -> Optional[np.ndarray]:
    """
    Crops a NumPy array representing an image according to the specified bounding box.

    Parameters
    ----------
    array : np.array
        The image as a NumPy array.
    bbox : Tuple[int, int, int, int]
        The bounding box to crop the image to, given as (w1, h1, w2, h2).
    min_width : int, optional
        The minimum allowable width for the cropped image. If the cropped width is smaller than this value,
        the function returns None. Default is 1.
    min_height : int, optional
        The minimum allowable height for the cropped image. If the cropped height is smaller than this value,
        the function returns None. Default is 1.

    Returns
    -------
    Optional[np.ndarray]
        The cropped image as a NumPy array, or None if the bounding box is invalid.
    """
    w1, h1, w2, h2 = bbox
    h1 = max(floor(h1), 0)
    h2 = min(ceil(h2), array.shape[0])
    w1 = max(floor(w1), 0)
    w2 = min(ceil(w2), array.shape[1])

    if (w2 - w1 < min_width) or (h2 - h1 < min_height):
        return None

    # Crop the image using the bounding box
    cropped = array[h1:h2, w1:w2]

    return cropped


def numpy_to_base64(array: np.ndarray) -> str:
    """
    Converts a NumPy array representing an image to a base64-encoded string.

    The function takes a NumPy array, converts it to a PIL image, and then encodes
    the image as a PNG in a base64 string format. The input array is expected to be in
    a format that can be converted to a valid image, such as having a shape of (H, W, C)
    where C is the number of channels (e.g., 3 for RGB).

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array. Must have a shape compatible with image data.

    Returns
    -------
    str
        The base64-encoded string representation of the input NumPy array as a PNG image.

    Raises
    ------
    ValueError
        If the input array cannot be converted into a valid image format.
    RuntimeError
        If there is an issue during the image conversion or base64 encoding process.

    Examples
    --------
    >>> array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> encoded_str = numpy_to_base64(array)
    >>> isinstance(encoded_str, str)
    True
    """
    # If the array represents a grayscale image, drop the redundant axis in
    # (h, w, 1). PIL.Image.fromarray() expects an array of form (h, w) if it's
    # a grayscale image.
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.squeeze(array, axis=2)

    # Check if the array is valid and can be converted to an image
    try:
        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(array.astype(np.uint8))
    except Exception as e:
        raise ValueError(f"Failed to convert NumPy array to image: {e}")

    try:
        # Convert the PIL image to a base64-encoded string
        with BytesIO() as buffer:
            pil_image.save(buffer, format="PNG")
            base64_img = bytetools.base64frombytes(buffer.getvalue())
    except Exception as e:
        raise RuntimeError(f"Failed to encode image to base64: {e}")

    return base64_img
