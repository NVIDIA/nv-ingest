# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import mimetypes
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import requests
from PIL import Image


def detect_mime_type(image_path: str) -> str:
    """Detect MIME type from file extension or image header."""
    mime, _ = mimetypes.guess_type(image_path)
    if mime:
        return mime
    # Fallback via PIL
    try:
        with Image.open(image_path) as im:
            fmt = (im.format or "PNG").upper()
        if fmt == "JPEG":
            return "image/jpeg"
        return f"image/{fmt.lower()}"
    except Exception:
        return "image/png"


essential_headers: Dict[str, str] = {
    "accept": "application/json",
    "content-type": "application/json",
}


def encode_image_file_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_file_to_data_url(image_path: str) -> str:
    b64 = encode_image_file_to_base64(image_path)
    mime = detect_mime_type(image_path)
    return f"data:{mime};base64,{b64}"


def load_image_numpy(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        return np.array(img)


def build_http_payload_from_data_url(data_url: str, batch_size: int) -> Dict:
    input_list = [{"type": "image_url", "url": data_url} for _ in range(max(1, batch_size))]
    return {"input": input_list}


def http_post_with_timing(url: str, payload: Dict, timeout: int, headers: Dict[str, str]) -> Tuple[object, float]:
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=timeout, headers=headers)
        return response.status_code, time.time() - start
    except requests.exceptions.Timeout:
        return "timeout", time.time() - start
    except Exception:
        return "error", time.time() - start


def yolox_grpc_infer_with_timing(
    endpoint: str,
    auth_token: str,
    image_np: np.ndarray,
    batch_size: int,
    timeout: int,
) -> Tuple[object, float]:
    """
    Perform a single Yolox inference over gRPC and measure elapsed time.

    Returns (status_code_or_label, elapsed_seconds)
    - 200 on success
    - "timeout" if elapsed exceeds timeout
    - "error" on exception
    """
    # Lazy import to avoid hard dependency at module import time and to allow sys.path tweaks
    try:
        # Ensure local repo api/src is importable when running from source tree
        here = os.path.dirname(os.path.abspath(__file__))
        api_src = os.path.abspath(os.path.join(here, "..", "..", "..", "api", "src"))
        if api_src not in sys.path:
            sys.path.insert(0, api_src)

        from nv_ingest_api.internal.primitives.nim.model_interface.yolox import (  # type: ignore
            YoloxPageElementsModelInterface,
            get_yolox_model_name,
        )
        from nv_ingest_api.util.nim import create_inference_client  # type: ignore
    except Exception:
        return "error", 0.0

    model_interface = YoloxPageElementsModelInterface()
    client = None
    start = time.time()
    try:
        # Create client for gRPC only
        client = create_inference_client(
            (endpoint, ""),
            model_interface,
            auth_token,
            infer_protocol="grpc",
            timeout=float(timeout),
        )
        model_name = get_yolox_model_name(endpoint, default_model_name="yolox_ensemble")

        data = {"images": [image_np for _ in range(max(1, batch_size))]}

        # Note: We cannot reliably enforce a deadline without wiring it through the client;
        # we treat overtime as timeout.
        _ = client.infer(
            data,
            model_name=model_name,
            max_batch_size=max(1, batch_size),
            trace_info=None,
            stage_name="concurrency_test",
            input_names=["INPUT_IMAGES", "THRESHOLDS"],
            dtypes=["BYTES", "FP32"],
            output_names=["OUTPUT"],
        )
        elapsed = time.time() - start
        if elapsed > timeout:
            return "timeout", elapsed
        return 200, elapsed
    except Exception:
        return "error", time.time() - start
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:
            pass
