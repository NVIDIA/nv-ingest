# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from morpheus.config import Config

from nv_ingest.schemas.infographic_extractor_schema import InfographicExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.image_processing.transforms import base64_to_numpy
from nv_ingest.util.nim.helpers import NimClient
from nv_ingest.util.nim.helpers import create_inference_client
from nv_ingest.util.nim.paddle import PaddleOCRModelInterface

logger = logging.getLogger(__name__)

PADDLE_MIN_WIDTH = 32
PADDLE_MIN_HEIGHT = 32


def _update_metadata(
    base64_images: List[str],
    paddle_client: NimClient,
    worker_pool_size: int = 8,  # Not currently used
    trace_info: Dict = None,
) -> List[Tuple[str, Tuple[Any, Any]]]:
    """
    Given a list of base64-encoded images, this function filters out images that do not meet the minimum
    size requirements and then calls the PaddleOCR model via paddle_client.infer to extract infographic data.

    For each base64-encoded image, the result is:
        (base64_image, (text_predictions, bounding_boxes))

    Images that do not meet the minimum size are skipped (resulting in ("", "") for that image).
    The paddle_client is expected to handle any necessary batching and concurrency.
    """
    logger.debug(f"Running infographic extraction using protocol {paddle_client.protocol}")

    # Initialize the results list in the same order as base64_images.
    results: List[Optional[Tuple[str, Tuple[Any, Any, Any]]]] = [("", None, None)] * len(base64_images)

    valid_images: List[str] = []
    valid_indices: List[int] = []

    # Pre-decode image dimensions and filter valid images.
    for i, img in enumerate(base64_images):
        array = base64_to_numpy(img)
        height, width = array.shape[0], array.shape[1]
        if width >= PADDLE_MIN_WIDTH and height >= PADDLE_MIN_HEIGHT:
            valid_images.append(img)
            valid_indices.append(i)
        else:
            # Image is too small; mark as skipped.
            results[i] = (img, None, None)

    # Prepare data payloads for both clients.
    data_paddle = {"base64_images": valid_images}

    _ = worker_pool_size

    try:
        paddle_results = paddle_client.infer(
            data=data_paddle,
            model_name="paddle",
            stage_name="infographic_data_extraction",
            max_batch_size=1 if paddle_client.protocol == "grpc" else 2,
            trace_info=trace_info,
        )
    except Exception as e:
        logger.error(f"Error calling paddle_client.infer: {e}", exc_info=True)
        raise

    if len(paddle_results) != len(valid_images):
        raise ValueError(f"Expected {len(valid_images)} paddle results, got {len(paddle_results)}")

    for idx, paddle_res in enumerate(paddle_results):
        original_index = valid_indices[idx]
        results[original_index] = (base64_images[original_index], paddle_res[0], paddle_res[1])

    return results


def _create_clients(
    paddle_endpoints: Tuple[str, str],
    paddle_protocol: str,
    auth_token: str,
) -> Tuple[NimClient, NimClient]:
    paddle_model_interface = PaddleOCRModelInterface()

    logger.debug(f"Inference protocols: paddle={paddle_protocol}")

    paddle_client = create_inference_client(
        endpoints=paddle_endpoints,
        model_interface=paddle_model_interface,
        auth_token=auth_token,
        infer_protocol=paddle_protocol,
    )

    return paddle_client


def _extract_infographic_data(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts infographic data from a DataFrame in a bulk fashion rather than row-by-row,
    following the chart extraction pattern.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which infographic data is to be extracted.
    task_props : Dict[str, Any]
        Dictionary containing task properties and configurations.
    validated_config : Any
        The validated configuration object for infographic extraction.
    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.
    """

    _ = task_props  # unused

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df.empty:
        return df, trace_info

    stage_config = validated_config.stage_config
    paddle_client = _create_clients(
        stage_config.paddle_endpoints,
        stage_config.paddle_infer_protocol,
        stage_config.auth_token,
    )

    try:
        # 1) Identify rows that meet criteria
        # (structured, subtype=infographic, table_metadata != None, content not empty)
        def meets_criteria(row):
            m = row.get("metadata", {})
            if not m:
                return False
            content_md = m.get("content_metadata", {})
            if (
                content_md.get("type") == "structured"
                and content_md.get("subtype") == "infographic"
                and m.get("table_metadata") is not None
                and m.get("content") not in [None, ""]
            ):
                return True
            return False

        mask = df.apply(meets_criteria, axis=1)
        valid_indices = df[mask].index.tolist()

        # If no rows meet the criteria, just return
        if not valid_indices:
            return df, {"trace_info": trace_info}

        # 2) Extract base64 images in the same order
        base64_images = []
        for idx in valid_indices:
            meta = df.at[idx, "metadata"]
            base64_images.append(meta["content"])

        # 3) Call our bulk _update_metadata to get all results
        bulk_results = _update_metadata(
            base64_images=base64_images,
            paddle_client=paddle_client,
            worker_pool_size=stage_config.workers_per_progress_engine,
            trace_info=trace_info,
        )

        # 4) Write the results (bounding_boxes, text_predictions) back
        for row_id, idx in enumerate(valid_indices):
            # unpack (base64_image, paddle_bounding boxes, paddle_text_predictions)
            _, _, text_predictions = bulk_results[row_id]
            table_content = " ".join(text_predictions) if text_predictions else None

            df.at[idx, "metadata"]["table_metadata"]["table_content"] = table_content

        return df, {"trace_info": trace_info}

    except Exception:
        logger.error("Error occurred while extracting infographic data.", exc_info=True)
        traceback.print_exc()
        raise
    finally:
        paddle_client.close()


def generate_infographic_extractor_stage(
    c: Config,
    stage_config: Dict[str, Any],
    task: str = "infographic_data_extract",
    task_desc: str = "infographic_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform infographic data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the infographic content extractor, passed as a dictionary
        validated against the `TableExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific infographic extraction process.
        Default is "infographic_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during infographic extraction.
        Default is "infographic_data_extraction".

    pe_count : int, optional
        The number of process engines to use for infographic data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles infographic data extraction
        from PDF content.
    """

    validated_config = InfographicExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_extract_infographic_data, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
