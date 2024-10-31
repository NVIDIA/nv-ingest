# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from math import ceil
from math import floor
from typing import Optional
from typing import Tuple

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from nv_ingest.util.converters import bytetools

DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1280


def pad_image(
    array: np.ndarray,
    target_width: int = DEFAULT_MAX_WIDTH,
    target_height: int = DEFAULT_MAX_HEIGHT,
    background_color: int = 255,
    dtype=np.uint8,
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
    canvas = background_color * np.ones((final_height, final_width, array.shape[2]), dtype=dtype)
    canvas[pad_height : pad_height + height, pad_width : pad_width + width] = array  # noqa: E203

    return canvas, (pad_width, pad_height)

def check_numpy_image_size(image: np.ndarray, min_height: int, min_width: int) -> bool:
    """
    Checks if the height and width of the image are larger than the specified minimum values.

    Parameters:
    image (np.ndarray): The image array (assumed to be in shape (H, W, C) or (H, W)).
    min_height (int): The minimum height required.
    min_width (int): The minimum width required.

    Returns:
    bool: True if the image dimensions are larger than or equal to the minimum size, False otherwise.
    """
    # Check if the image has at least 2 dimensions
    if image.ndim < 2:
        raise ValueError("The input array does not have sufficient dimensions for an image.")

    height, width = image.shape[:2]
    return height >= min_height and width >= min_width

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


def normalize_image(
    array: np.ndarray,
    r_mean: float = 0.485,
    g_mean: float = 0.456,
    b_mean: float = 0.406,
    r_std: float = 0.229,
    g_std: float = 0.224,
    b_std: float = 0.225,
) -> np.ndarray:
    """
    Normalizes an RGB image by applying a mean and standard deviation to each channel.

    Parameters:
    ----------
    array : np.ndarray
        The input image array, which can be either grayscale or RGB. The image should have a shape of
        (height, width, 3) for RGB images, or (height, width) or (height, width, 1) for grayscale images.
        If a grayscale image is provided, it will be converted to RGB format by repeating the grayscale values
        across all three channels (R, G, B).
    r_mean : float, optional
        The mean to be subtracted from the red channel (default is 0.485).
    g_mean : float, optional
        The mean to be subtracted from the green channel (default is 0.456).
    b_mean : float, optional
        The mean to be subtracted from the blue channel (default is 0.406).
    r_std : float, optional
        The standard deviation to divide the red channel by (default is 0.229).
    g_std : float, optional
        The standard deviation to divide the green channel by (default is 0.224).
    b_std : float, optional
        The standard deviation to divide the blue channel by (default is 0.225).

    Returns:
    -------
    np.ndarray
        A normalized image array with the same shape as the input, where the RGB channels have been normalized
        by the given means and standard deviations.

    Notes:
    -----
    The input pixel values should be in the range [0, 255], and the function scales these values to [0, 1]
    before applying normalization.

    If the input image is grayscale, it is converted to an RGB image by duplicating the grayscale values
    across the three color channels.
    """
    # If the input is a grayscale image with shape (height, width) or (height, width, 1),
    # convert it to RGB with shape (height, width, 3).
    if array.ndim == 2 or array.shape[2] == 1:
        array = np.dstack((array, 255 * np.ones_like(array), 255 * np.ones_like(array)))

    height, width = array.shape[:2]

    mean = np.array([r_mean, g_mean, b_mean]).reshape((1, 1, 3)).astype(np.float32)
    std = np.array([r_std, g_std, b_std]).reshape((1, 1, 3)).astype(np.float32)
    output_array = (array.astype("float32") / 255.0 - mean) / std

    return output_array


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


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert a base64-encoded image string to a NumPy array.

    Parameters
    ----------
    base64_string : str
        Base64-encoded string representing an image.

    Returns
    -------
    numpy.ndarray
        NumPy array representation of the decoded image.

    Raises
    ------
    ValueError
        If the base64 string is invalid or cannot be decoded into an image.
    ImportError
        If required libraries are not installed.

    Examples
    --------
    >>> base64_str = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD...'
    >>> img_array = base64_to_numpy(base64_str)
    """
    try:
        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
    except (base64.binascii.Error, ValueError) as e:
        raise ValueError("Invalid base64 string") from e

    try:
        # Convert the bytes into a BytesIO object
        image_bytes = BytesIO(image_data)

        # Open the image using PIL
        image = Image.open(image_bytes)
        image.load()
    except UnidentifiedImageError as e:
        raise ValueError("Unable to decode image from base64 string") from e

    # Convert the image to a NumPy array
    image_array = np.array(image)

    return image_array
