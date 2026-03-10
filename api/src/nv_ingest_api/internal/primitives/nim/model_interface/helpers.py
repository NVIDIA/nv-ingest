# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import backoff
import cv2
import numpy as np
import requests

from nv_ingest_api.internal.primitives.nim.model_interface.decorators import multiprocessing_cache
from nv_ingest_api.util.image_processing.transforms import pad_image, normalize_image
from nv_ingest_api.util.string_processing import generate_url, remove_url_endpoints

cv2.setNumThreads(1)
logger = logging.getLogger(__name__)


def preprocess_image_for_paddle(array: np.ndarray, image_max_dimension: int = 960) -> np.ndarray:
    """
    Preprocesses an input image to be suitable for use with PaddleOCR by resizing, normalizing, padding,
    and transposing it into the required format.

    This function is intended for preprocessing images to be passed as input to PaddleOCR using GRPC.
    It is not necessary when using the HTTP endpoint.

    Steps:
    -----
    1. Resizes the image while maintaining aspect ratio such that its largest dimension is scaled to 960 pixels.
    2. Normalizes the image using the `normalize_image` function.
    3. Pads the image to ensure both its height and width are multiples of 32, as required by PaddleOCR.
    4. Transposes the image from (height, width, channel) to (channel, height, width), the format expected by PaddleOCR.

    Parameters:
    ----------
    array : np.ndarray
        The input image array of shape (height, width, channels). It should have pixel values in the range [0, 255].

    Returns:
    -------
    np.ndarray
        A preprocessed image with the shape (channels, height, width) and normalized pixel values.
        The image will be padded to have dimensions that are multiples of 32, with the padding color set to 0.

    Notes:
    -----
    - The image is resized so that its largest dimension becomes 960 pixels, maintaining the aspect ratio.
    - After normalization, the image is padded to the nearest multiple of 32 in both dimensions, which is
      a requirement for PaddleOCR.
    - The normalized pixel values are scaled between 0 and 1 before padding and transposing the image.
    """
    height, width = array.shape[:2]
    scale_factor = image_max_dimension / max(height, width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized = cv2.resize(array, (new_width, new_height))

    normalized = normalize_image(resized)

    # PaddleOCR NIM (GRPC) requires input shapes to be multiples of 32.
    new_height = (normalized.shape[0] + 31) // 32 * 32
    new_width = (normalized.shape[1] + 31) // 32 * 32
    padded, (pad_width, pad_height) = pad_image(
        normalized, target_height=new_height, target_width=new_width, background_color=0, dtype=np.float32
    )

    # PaddleOCR NIM (GRPC) requires input to be (channel, height, width).
    transposed = padded.transpose((2, 0, 1))

    # Metadata can used for inverting transformations on the resulting bounding boxes.
    metadata = {
        "original_height": height,
        "original_width": width,
        "scale_factor": scale_factor,
        "new_height": transposed.shape[1],
        "new_width": transposed.shape[2],
        "pad_height": pad_height,
        "pad_width": pad_width,
    }

    return transposed, metadata


def preprocess_image_for_ocr(
    array: np.ndarray,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    pad_how: str = "bottom_right",
    normalize: bool = False,
    channel_first: bool = False,
) -> np.ndarray:
    """
    Preprocesses an input image to be suitable for use with NemoRetriever-OCR.

    This function is intended for preprocessing images to be passed as input to NemoRetriever-OCR using GRPC.
    It is not necessary when using the HTTP endpoint.

    Parameters:
    ----------
    array : np.ndarray
        The input image array of shape (height, width, channels). It should have pixel values in the range [0, 255].

    Returns:
    -------
    np.ndarray
        A preprocessed image with the shape (channels, height, width).
    """
    height, width = array.shape[:2]

    if target_height is None:
        target_height = height

    if target_width is None:
        target_width = width

    padded, (pad_width, pad_height) = pad_image(
        array,
        target_height=target_height,
        target_width=target_width,
        background_color=255,
        dtype=np.float32,
        how=pad_how,
    )

    if normalize:
        padded = padded / 255.0

    if channel_first:
        # NemoRetriever-OCR NIM (GRPC) requires input to be (channel, height, width).
        padded = padded.transpose((2, 0, 1))

    # Metadata can used for inverting transformations on the resulting bounding boxes.
    metadata = {
        "original_height": height,
        "original_width": width,
        "new_height": target_height,
        "new_width": target_width,
        "pad_height": pad_height,
        "pad_width": pad_width,
    }

    return padded, metadata


def is_ready(http_endpoint: str, ready_endpoint: str) -> bool:
    """
    Check if the server at the given endpoint is ready.

    Parameters
    ----------
    http_endpoint : str
        The HTTP endpoint of the server.
    ready_endpoint : str
        The specific ready-check endpoint.

    Returns
    -------
    bool
        True if the server is ready, False otherwise.
    """

    # IF the url is empty or None that means the service was not configured
    # and is therefore automatically marked as "ready"
    if http_endpoint is None or http_endpoint == "":
        return True

    # If the url is for build.nvidia.com, it is automatically assumed "ready"
    if "ai.api.nvidia.com" in http_endpoint:
        return True

    url = generate_url(http_endpoint)
    url = remove_url_endpoints(url)

    if not ready_endpoint.startswith("/") and not url.endswith("/"):
        ready_endpoint = "/" + ready_endpoint

    url = url + ready_endpoint

    # Call the ready endpoint of the NIM
    try:
        # Use a short timeout to prevent long hanging calls. 5 seconds seems resonable
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            # The NIM is saying it is ready to serve
            return True
        elif resp.status_code == 503:
            # NIM is explicitly saying it is not ready.
            return False
        else:
            # Any other code is confusing. We should log it with a warning
            # as it could be something that might hold up ready state
            logger.warning(f"'{url}' HTTP Status: {resp.status_code} - Response Payload: {resp.json()}")
            return False
    except requests.HTTPError as http_err:
        logger.warning(f"'{url}' produced a HTTP error: {http_err}")
        return False
    except requests.Timeout:
        logger.warning(f"'{url}' request timed out")
        return False
    except ConnectionError:
        logger.warning(f"A connection error for '{url}' occurred")
        return False
    except requests.RequestException as err:
        logger.warning(f"An error occurred: {err} for '{url}'")
        return False
    except Exception as ex:
        # Don't let anything squeeze by
        logger.warning(f"Exception: {ex}")
        return False


def _query_metadata(
    http_endpoint: str,
    field_name: str,
    default_value: str,
    retry_value: str = "",
    metadata_endpoint: str = "/v1/metadata",
) -> str:
    if (http_endpoint is None) or (http_endpoint == ""):
        return default_value

    url = generate_url(http_endpoint)
    url = remove_url_endpoints(url)

    if not metadata_endpoint.startswith("/") and not url.endswith("/"):
        metadata_endpoint = "/" + metadata_endpoint

    url = url + metadata_endpoint

    # Call the metadata endpoint of the NIM
    try:
        # Use a short timeout to prevent long hanging calls. 5 seconds seems reasonable
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            field_value = resp.json().get(field_name, "")
            if field_value:
                return field_value
            else:
                # If the field is empty, retry
                logger.warning(f"No {field_name} field in response from '{url}'. Retrying.")
                return retry_value
        else:
            # Any other code is confusing. We should log it with a warning
            logger.warning(f"'{url}' HTTP Status: {resp.status_code} - Response Payload: {resp.text}")
            return retry_value
    except requests.HTTPError as http_err:
        logger.warning(f"'{url}' produced a HTTP error: {http_err}")
        return retry_value
    except requests.Timeout:
        logger.warning(f"'{url}' request timed out")
        return retry_value
    except ConnectionError:
        logger.warning(f"A connection error for '{url}' occurred")
        return retry_value
    except requests.RequestException as err:
        logger.warning(f"An error occurred: {err} for '{url}'")
        return retry_value
    except Exception as ex:
        # Don't let anything squeeze by
        logger.warning(f"Exception: {ex}")
        return retry_value


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_version(http_endpoint: str, metadata_endpoint: str = "/v1/metadata", version_field: str = "version") -> str:
    """
    Get the version of the server from its metadata endpoint.

    Parameters
    ----------
    http_endpoint : str
        The HTTP endpoint of the server.
    metadata_endpoint : str, optional
        The metadata endpoint to query (default: "/v1/metadata").
    version_field : str, optional
        The field containing the version in the response (default: "version").

    Returns
    -------
    str
        The version of the server, or an empty string if unavailable.
    """
    default_version = "1.0.0"

    # TODO: Need a way to match NIM version to API versions.
    if "ai.api.nvidia.com" in http_endpoint or "api.nvcf.nvidia.com" in http_endpoint:
        return default_version

    return _query_metadata(
        http_endpoint,
        field_name=version_field,
        default_value=default_version,
    )


@multiprocessing_cache(max_calls=100)  # Cache results first to avoid redundant retries from backoff
@backoff.on_predicate(backoff.expo, max_time=30)
def get_model_name(
    http_endpoint: str,
    default_model_name,
    metadata_endpoint: str = "/v1/metadata",
    model_info_field: str = "modelInfo",
) -> str:
    """
    Get the model name of the server from its metadata endpoint.

    Parameters
    ----------
    http_endpoint : str
        The HTTP endpoint of the server.
    metadata_endpoint : str, optional
        The metadata endpoint to query (default: "/v1/metadata").
    model_info_field : str, optional
        The field containing the model info in the response (default: "modelInfo").

    Returns
    -------
    str
        The model name of the server, or an empty string if unavailable.
    """
    if "ai.api.nvidia.com" in http_endpoint:
        return http_endpoint.strip("/").strip("/chat/completions").split("/")[-1]

    if "api.nvcf.nvidia.com" in http_endpoint:
        return default_model_name

    model_info = _query_metadata(
        http_endpoint,
        field_name=model_info_field,
        default_value={"shortName": default_model_name},
    )
    short_name = model_info[0].get("shortName", default_model_name)
    model_name = short_name.split(":")[0]

    return model_name
