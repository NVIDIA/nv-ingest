# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from math import ceil
from math import floor
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from nv_ingest_api.util.converters import bytetools

# Configure OpenCV to use a single thread for image processing
cv2.setNumThreads(1)
DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1280

# Workaround for PIL.Image.DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


def _resize_image_opencv(
    array: np.ndarray, target_size: Tuple[int, int], interpolation=cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Resizes a NumPy array representing an image using OpenCV.

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array.
    target_size : Tuple[int, int]
        The target size as (width, height).
    interpolation : int, optional
        OpenCV interpolation method. Defaults to cv2.INTER_LANCZOS4.

    Returns
    -------
    np.ndarray
        The resized image as a NumPy array.
    """
    return cv2.resize(array, target_size, interpolation=interpolation)


def rgba_to_rgb_white_bg(rgba_image):
    """
    Convert RGBA image to RGB by blending with a white background.

    This function properly handles transparency by alpha-blending transparent
    and semi-transparent pixels with a white background, producing visually
    accurate results that match how the image would appear when displayed.

    Parameters
    ----------
    rgba_image : numpy.ndarray
        Input image array with shape (height, width, 4) where the channels
        are Red, Green, Blue, Alpha. Alpha values can be in range [0, 1]
        (float) or [0, 255] (uint8).

    Returns
    -------
    numpy.ndarray
        RGB image array with shape (height, width, 3) and dtype uint8.
        Values are in range [0, 255] representing Red, Green, Blue channels.

    Notes
    -----
    The alpha blending formula used is:
        RGB_out = RGB_in * alpha + background * (1 - alpha)

    Where background is white (255, 255, 255).

    For pixels with alpha = 1.0 (fully opaque), the original RGB values
    are preserved. For pixels with alpha = 0.0 (fully transparent), the
    result is white. Semi-transparent pixels are blended proportionally.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a sample RGBA image with some transparency
    >>> rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    >>> rgb = rgba_to_rgb_white_bg(rgba)
    >>> print(rgb.shape)  # (100, 100, 3)
    >>> print(rgb.dtype)  # uint8

    >>> # Example with float alpha values [0, 1]
    >>> rgba_float = np.random.rand(50, 50, 4).astype(np.float32)
    >>> rgb_float = rgba_to_rgb_white_bg(rgba_float)
    >>> print(rgb_float.dtype)  # uint8
    """
    # Extract RGB and alpha channels
    rgb = rgba_image[:, :, :3]  # RGB channels (H, W, 3)
    alpha = rgba_image[:, :, 3:4]  # Alpha channel (H, W, 1)

    # Normalize alpha to [0, 1] range if it's in [0, 255] range
    if alpha.max() > 1.0:
        alpha = alpha / 255.0

    # Alpha blend with white background using the formula:
    # result = foreground * alpha + background * (1 - alpha)
    rgb_image = rgb * alpha + 255 * (1 - alpha)

    # Convert to uint8 format for standard image representation
    return rgb_image.astype(np.uint8)


def scale_image_to_encoding_size(
    base64_image: str, max_base64_size: int = 180_000, initial_reduction: float = 0.9, format: str = "PNG", **kwargs
) -> Tuple[str, Tuple[int, int]]:
    """
    Decodes a base64-encoded image, resizes it if needed, and re-encodes it as base64.
    Ensures the final image size is within the specified limit.

    Parameters
    ----------
    base64_image : str
        Base64-encoded image string.
    max_base64_size : int, optional
        Maximum allowable size for the base64-encoded image, by default 180,000 characters.
    initial_reduction : float, optional
        Initial reduction step for resizing, by default 0.9.
    format : str, optional
        The image format to use for encoding. Supported formats are "PNG" and "JPEG".
        Defaults to "PNG".
    **kwargs
        Additional keyword arguments passed to the format-specific encoding function.
        For JPEG: quality (int, default=100) - JPEG quality (1-100).
        For PNG: compression (int, default=3) - PNG compression level (0-9).

    Returns
    -------
    Tuple[str, Tuple[int, int]]
        A tuple containing:
        - Base64-encoded image string in the specified format, resized if necessary.
        - The new size as a tuple (width, height).

    Raises
    ------
    Exception
        If the image cannot be resized below the specified max_base64_size.
    """
    try:
        # Decode the base64 image using OpenCV (returns RGB format)
        img_array = base64_to_numpy(base64_image)

        # Initial image size (height, width, channels) -> (width, height)
        original_size = (img_array.shape[1], img_array.shape[0])

        # Check initial size
        if len(base64_image) <= max_base64_size:
            return numpy_to_base64(img_array, format=format, **kwargs), original_size

        # Initial reduction step
        reduction_step = initial_reduction
        new_size = original_size
        current_img = img_array.copy()
        original_width, original_height = original_size

        while len(base64_image) > max_base64_size:
            new_size = (int(original_width * reduction_step), int(original_height * reduction_step))
            if new_size[0] < 1 or new_size[1] < 1:
                raise ValueError("Image cannot be resized further without becoming too small.")

            # Resize the image using OpenCV
            current_img = _resize_image_opencv(img_array, new_size)

            # Re-encode as base64 using the specified format
            base64_image = numpy_to_base64(current_img, format=format, **kwargs)

            # Adjust the reduction step if necessary
            if len(base64_image) > max_base64_size:
                reduction_step *= 0.95  # Reduce size further if needed

        return base64_image, new_size

    except Exception as e:
        logger.error(f"Error resizing the image: {e}")
        raise


def _detect_base64_image_format(base64_string: str) -> Optional[str]:
    """
    Detects the format of a base64-encoded image using Pillow.

    Parameters
    ----------
    base64_string : str
        Base64-encoded image string.

    Returns
    -------
    The detected format ("PNG", "JPEG", "UNKNOWN")
    """
    try:
        image_bytes = bytetools.bytesfrombase64(base64_string)
    except Exception as e:
        logger.error(f"Invalid base64 string: {e}")
        raise ValueError(f"Invalid base64 string: {e}") from e

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.format.upper()
    except ImportError:
        raise ImportError("Pillow library not available")
    except Exception as e:
        logger.error(f"Error detecting image format: {e}")
        return "UNKNOWN"


def ensure_base64_format(base64_image: str, target_format: str = "PNG", **kwargs) -> str:
    """
    Ensures the given base64-encoded image is in the specified format. Converts if necessary.
    Skips conversion if the image is already in the target format.

    Parameters
    ----------
    base64_image : str
        Base64-encoded image string.
    target_format : str, optional
        The target image format. Supported formats are "PNG", "JPEG"/"JPG". Defaults to "PNG".
    **kwargs
        Additional keyword arguments passed to the format-specific encoding function.
        For JPEG: quality (int, default=100) - JPEG quality (1-100).
        For PNG: compression (int, default=3) - PNG compression level (0-9).

    Returns
    -------
    str
        Base64-encoded image string in the specified format.

    Raises
    ------
    ValueError
        If there is an error during format conversion or if an unsupported format is provided.
    """
    # Quick format normalization
    target_format = target_format.upper().strip()
    if target_format == "JPG":
        target_format = "JPEG"

    current_format = _detect_base64_image_format(base64_image)
    if current_format == "UNKNOWN":
        raise ValueError(
            f"Unable to decode image from base64 string: {base64_image}, because current format could not be detected."
        )
    if current_format == target_format:
        logger.debug(f"Image already in {target_format} format, skipping conversion")
        return base64_image

    try:
        # Decode the base64 image using OpenCV (returns RGB format)
        img_array = base64_to_numpy(base64_image)
        # Re-encode in the target format
        return numpy_to_base64(img_array, format=target_format, **kwargs)
    except ImportError as e:
        raise e
    except Exception as e:
        logger.error(f"Error converting image to {target_format} format: {e}")
        raise ValueError(f"Failed to convert image to {target_format} format: {e}") from e


def pad_image(
    array: np.ndarray,
    target_width: int = DEFAULT_MAX_WIDTH,
    target_height: int = DEFAULT_MAX_HEIGHT,
    background_color: int = 255,
    dtype=np.uint8,
    how: str = "center",
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pads a NumPy array representing an image to the specified target dimensions.

    If the target dimensions are smaller than the image dimensions, no padding will be applied
    in that dimension. If the target dimensions are larger, the image will be centered within the
    canvas of the specified target size, with the remaining space filled with white padding.

    The padding can be done around the center (how="center"), or to the bottom right (how="bottom_right").

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array of shape (H, W, C).
    target_width : int, optional
        The desired target width of the padded image. Defaults to DEFAULT_MAX_WIDTH.
    target_height : int, optional
        The desired target height of the padded image. Defaults to DEFAULT_MAX_HEIGHT.
    how : str, optional
        The method to pad the image. Defaults to "center".

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

    # Determine final canvas size (may be equal to original if target is smaller)
    final_height = max(height, target_height)
    final_width = max(width, target_width)

    # Create the canvas and place the original image on it
    canvas = background_color * np.ones((final_height, final_width, array.shape[2]), dtype=dtype)

    # Determine the padding needed, if any, while ensuring no padding is applied if the target is smaller
    if how == "center":
        pad_height = max((target_height - height) // 2, 0)
        pad_width = max((target_width - width) // 2, 0)

        canvas[pad_height : pad_height + height, pad_width : pad_width + width] = array  # noqa: E203
    elif how == "bottom_right":
        pad_height, pad_width = 0, 0

        canvas[:height, :width] = array  # noqa: E203

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


def _preprocess_numpy_array(array: np.ndarray) -> np.ndarray:
    """
    Preprocesses a NumPy array for image encoding by ensuring proper format and data type.
    Also handles color space conversion for OpenCV encoding.

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The preprocessed array in uint8 format, ready for OpenCV encoding (BGR color order for color images).

    Raises
    ------
    ValueError
        If the input array cannot be converted into a valid image format.
    """
    # Check if the array is valid and can be converted to an image
    try:
        # If the array represents a grayscale image, drop the redundant axis in
        # (h, w, 1). cv2 expects (h, w) for grayscale.
        if array.ndim == 3 and array.shape[2] == 1:
            array = np.squeeze(array, axis=2)

        # Ensure uint8 data type
        processed_array = array.astype(np.uint8)

        # OpenCV uses BGR color order, so convert RGB to BGR if needed
        if processed_array.ndim == 3 and processed_array.shape[2] == 3:
            # Assume input is RGB and convert to BGR for OpenCV
            processed_array = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)

        return processed_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess NumPy array for image encoding: {e}")


def _encode_opencv_jpeg(array: np.ndarray, *, quality: int = 100) -> bytes:
    """NumPy array -> JPEG bytes using OpenCV."""
    ok, buf = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _encode_opencv_png(array: np.ndarray, *, compression: int = 6) -> bytes:
    """NumPy array -> PNG bytes using OpenCV"""
    encode_params = [
        cv2.IMWRITE_PNG_COMPRESSION,
        compression,
        cv2.IMWRITE_PNG_STRATEGY,
        cv2.IMWRITE_PNG_STRATEGY_DEFAULT,
    ]
    ok, buf = cv2.imencode(".png", array, encode_params)
    if not ok:
        raise RuntimeError("cv2.imencode(.png) failed")
    return buf.tobytes()


def numpy_to_base64_png(array: np.ndarray) -> str:
    """
    Converts a preprocessed NumPy array representing an image to a base64-encoded PNG string using OpenCV.

    Parameters
    ----------
    array : np.ndarray
        The preprocessed input image as a NumPy array. Must have a shape compatible with image data.

    Returns
    -------
    str
        The base64-encoded PNG string representation of the input NumPy array.

    Raises
    ------
    RuntimeError
        If there is an issue during the image conversion or base64 encoding process.
    """
    try:
        # Encode to PNG bytes using OpenCV
        png_bytes = _encode_opencv_png(array)

        # Convert to base64
        base64_img = bytetools.base64frombytes(png_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to encode image to base64 PNG: {e}")

    return base64_img


def numpy_to_base64_jpeg(array: np.ndarray, quality: int = 100) -> str:
    """
    Converts a preprocessed NumPy array representing an image to a base64-encoded JPEG string using OpenCV.

    Parameters
    ----------
    array : np.ndarray
        The preprocessed input image as a NumPy array. Must have a shape compatible with image data.
    quality : int, optional
        JPEG quality (1-100), by default 100. Higher values mean better quality but larger file size.

    Returns
    -------
    str
        The base64-encoded JPEG string representation of the input NumPy array.

    Raises
    ------
    RuntimeError
        If there is an issue during the image conversion or base64 encoding process.
    """
    try:
        # Encode to JPEG bytes using OpenCV
        jpeg_bytes = _encode_opencv_jpeg(array, quality=quality)

        # Convert to base64
        base64_img = bytetools.base64frombytes(jpeg_bytes)
    except Exception as e:
        raise RuntimeError(f"Failed to encode image to base64 JPEG: {e}")

    return base64_img


def numpy_to_base64(array: np.ndarray, format: str = "PNG", **kwargs) -> str:
    """
    Converts a NumPy array representing an image to a base64-encoded string.

    The function takes a NumPy array, preprocesses it, and then encodes
    the image in the specified format as a base64 string. The input array is expected
    to be in a format that can be converted to a valid image, such as having a shape
    of (H, W, C) where C is the number of channels (e.g., 3 for RGB).

    Parameters
    ----------
    array : np.ndarray
        The input image as a NumPy array. Must have a shape compatible with image data.
    format : str, optional
        The image format to use for encoding. Supported formats are "PNG" and "JPEG".
        Defaults to "PNG".
    **kwargs
        Additional keyword arguments passed to the format-specific encoding function.
        For JPEG: quality (int, default=100) - JPEG quality (1-100).

    Returns
    -------
    str
        The base64-encoded string representation of the input NumPy array in the specified format.

    Raises
    ------
    ValueError
        If the input array cannot be converted into a valid image format, or if an
        unsupported format is specified.
    RuntimeError
        If there is an issue during the image conversion or base64 encoding process.

    Examples
    --------
    >>> array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    >>> encoded_str = numpy_to_base64(array, format="PNG")
    >>> isinstance(encoded_str, str)
    True
    >>> encoded_str_jpeg = numpy_to_base64(array, format="JPEG", quality=90)
    >>> isinstance(encoded_str_jpeg, str)
    True
    """
    # Centralized preprocessing of the numpy array
    processed_array = _preprocess_numpy_array(array)

    # Quick format normalization
    format = format.upper().strip()
    if format == "JPG":
        format = "JPEG"

    if format == "PNG":
        return numpy_to_base64_png(processed_array)
    elif format == "JPEG":
        quality = kwargs.get("quality", 100)
        return numpy_to_base64_jpeg(processed_array, quality=quality)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'PNG' and 'JPEG'.")


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert a base64-encoded image string to a NumPy array using OpenCV.
    Returns images in RGB format for consistency.

    Parameters
    ----------
    base64_string : str
        Base64-encoded string representing an image.

    Returns
    -------
    numpy.ndarray
        NumPy array representation of the decoded image in RGB format (for color images).
        Grayscale images are returned as-is.

    Raises
    ------
    ValueError
        If the base64 string is invalid or cannot be decoded into an image.

    Examples
    --------
    >>> base64_str = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBD...'
    >>> img_array = base64_to_numpy(base64_str)
    >>> # img_array is now in RGB format (for color images)
    """
    try:
        # Decode the base64 string to bytes using bytetools
        image_bytes = bytetools.bytesfrombase64(base64_string)
    except Exception as e:
        raise ValueError("Invalid base64 string") from e

    # Create numpy buffer from bytes and decode using OpenCV
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV failed to decode image")

        # Convert 4 channel to 3 channel if necessary
        if img.shape[2] == 4:
            img = rgba_to_rgb_white_bg(img)

        # Convert BGR to RGB for consistent processing (OpenCV loads as BGR)
        # Only convert if it's a 3-channel color image
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        raise
    except Exception as e:
        raise ValueError("Unable to decode image from base64 string") from e

    # Convert to numpy array
    img = np.array(img)
    # Assert that 3-channel images are in RGB format after conversion
    assert img.ndim <= 3, f"Image has unexpected number of dimensions: {img.ndim}"
    assert img.ndim != 3 or img.shape[2] == 3, f"3-channel image should have 3 channels, got: {img.shape[2]}"

    return img


def scale_numpy_image(
    img_arr: np.ndarray, scale_tuple: Optional[Tuple[int, int]] = None, interpolation=Image.LANCZOS
) -> np.ndarray:
    """
    Scales a NumPy image array using OpenCV with aspect ratio preservation.

    This function provides OpenCV-based image scaling that mimics PIL's thumbnail behavior
    by maintaining aspect ratio and scaling to fit within the specified dimensions.

    Parameters
    ----------
    img_arr : np.ndarray
        The input image as a NumPy array.
    scale_tuple : Optional[Tuple[int, int]], optional
        A tuple (width, height) to resize the image to. If provided, the image
        will be resized to fit within these dimensions while maintaining aspect ratio
        (similar to PIL's thumbnail method). Defaults to None.
    interpolation : int, optional
        OpenCV interpolation method. Defaults to cv2.INTER_LANCZOS4.

    Returns
    -------
    np.ndarray
        A NumPy array representing the scaled image data.
    """
    # Apply scaling using OpenCV if specified
    # Using PIL for scaling as CV2 seems to lead to different results
    # TODO: Remove when we move to YOLOX Ensemble Models
    if scale_tuple:
        image = Image.fromarray(img_arr)
        image.thumbnail(scale_tuple, interpolation)
        img_arr = np.array(image)
    # Ensure we return a copy
    return img_arr.copy()


def base64_to_disk(base64_string: str, output_path: str) -> bool:
    """
    Write base64-encoded image data directly to disk without conversion.

    This function performs efficient base64 decoding and direct file writing,
    preserving the original image format without unnecessary decode/encode cycles.
    Used as the foundation for higher-level image saving operations.

    Parameters
    ----------
    base64_string : str
        Base64-encoded image data. May include data URL prefix.
    output_path : str
        Path where the image should be saved.

    Returns
    -------
    bool
        True if successful, False otherwise.

    Examples
    --------
    >>> success = base64_to_disk(image_b64, "/path/to/output.jpeg")
    >>> if success:
    ...     print("Image saved successfully")
    """
    try:
        # Validate input
        if not base64_string or not base64_string.strip():
            return False

        # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode and write directly using bytetools (consistent with rest of codebase)
        image_bytes = bytetools.bytesfrombase64(base64_string)

        # Validate we actually have image data
        if not image_bytes:
            return False

        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return True

    except Exception as e:
        logger.error(f"Failed to write base64 image to disk: {e}")
        return False


def save_image_to_disk(base64_content: str, output_path: str, target_format: str = "auto", **kwargs) -> bool:
    """
    Save base64 image to disk with optional format conversion.

    This function provides a high-level interface for saving images that combines
    format conversion capabilities with efficient disk writing. It automatically
    chooses between direct writing (when no conversion needed) and format conversion
    to optimize performance while maintaining flexibility.

    Parameters
    ----------
    base64_content : str
        Base64-encoded image data.
    output_path : str
        Path where the image should be saved.
    target_format : str, optional
        Target format ("PNG", "JPEG", "auto"). Default is "auto" (preserve original).
        Use "auto" to preserve the original format for maximum speed.
    **kwargs
        Additional arguments passed to ensure_base64_format() for conversion.
        For JPEG: quality (int, default=100) - JPEG quality (1-100).
        For PNG: compression (int, default=3) - PNG compression level (0-9).

    Returns
    -------
    bool
        True if successful, False otherwise.

    Examples
    --------
    >>> # Preserve original format (fastest)
    >>> success = save_image_to_disk(image_b64, "/path/to/output.jpeg", "auto")
    >>>
    >>> # Convert to JPEG with specific quality
    >>> success = save_image_to_disk(image_b64, "/path/to/output.jpeg", "JPEG", quality=85)
    """
    try:
        # Quick format normalization
        target_format = target_format.lower().strip()
        if target_format in ["jpg"]:
            target_format = "jpeg"

        # Handle format conversion if needed
        if target_format == "auto":
            # Preserve original format - no conversion needed
            formatted_b64 = base64_content
        else:
            # Use API's smart format conversion
            formatted_b64 = ensure_base64_format(base64_content, target_format, **kwargs)

        # Direct write - no round trips
        return base64_to_disk(formatted_b64, output_path)

    except Exception as e:
        logger.error(f"Failed to save image to disk: {e}")
        return False
