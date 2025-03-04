# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import create_audio_inference_client

logger = logging.getLogger(__name__)


def _update_audio_metadata(row: pd.Series, audio_client: Any, trace_info: Dict) -> Dict:
    """
    Modifies the metadata of a row if the conditions for table extraction are met.

    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing metadata for the audio extraction.

    audio_client : Any
        The client used to call the audio inference model.

    trace_info : Dict
        Trace information used for logging or debugging.

    Returns
    -------
    Dict
        The modified metadata if conditions are met, otherwise the original metadata.

    Raises
    ------
    ValueError
        If critical information (such as metadata) is missing from the row.
    """

    metadata = row.get("metadata")

    if metadata is None:
        logger.error("Row does not contain 'metadata'.")
        raise ValueError("Row does not contain 'metadata'.")

    base64_audio = metadata.pop("content")
    content_metadata = metadata.get("content_metadata", {})

    # Only modify if content type is audio
    if (content_metadata.get("type") != "audio") and (base64_audio in (None, "")):
        return metadata

    # Modify audio metadata with the result from the inference model
    try:
        audio_result = audio_client.infer(
            base64_audio,
            model_name="parakeet",
            trace_info=trace_info,  # traceable_func arg
            stage_name="audio_extraction",
        )
        metadata["audio_metadata"] = {"audio_transcript": audio_result}
    except Exception as e:
        logger.exception(f"Unhandled error calling audio inference model: {e}", exc_info=True)

        raise

    return metadata


def _transcribe_audio(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts audio data from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which audio data is to be extracted.

    task_props : Dict[str, Any]
        Dictionary containing task properties and configurations.

    validated_config : Any
        The validated configuration object for audio extraction.

    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame and the trace information.

    Raises
    ------
    Exception
        If any error occurs during the audio data extraction process.
    """
    logger.debug(f"Entering audio extraction stage with {len(df)} rows.")

    extract_params = task_props.get("params", {}).get("extract_audio_params", {})
    stage_config = validated_config.audio_extraction_config

    grpc_endpoint = extract_params.get("grpc_endpoint") or stage_config.audio_endpoints[0]
    http_endpoint = extract_params.get("http_endpoint") or stage_config.audio_endpoints[1]
    infer_protocol = extract_params.get("infer_protocol") or stage_config.audio_infer_protocol
    auth_token = extract_params.get("auth_token") or stage_config.auth_token
    auth_metadata = extract_params.get("auth_metadata") or stage_config.auth_metadata
    use_ssl = extract_params.get("use_ssl") or stage_config.use_ssl
    ssl_cert = extract_params.get("ssl_cert") or stage_config.ssl_cert

    parakeet_client = create_audio_inference_client(
        (grpc_endpoint, http_endpoint),
        infer_protocol=infer_protocol,
        auth_token=auth_token,
        auth_metadata=auth_metadata,
        use_ssl=use_ssl,
        ssl_cert=ssl_cert,
    )

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_audio_metadata, axis=1, args=(parakeet_client, trace_info))

        return df, trace_info

    except Exception as e:
        logger.exception(f"Error occurred while extracting audio data: {e}", exc_info=True)

        raise
