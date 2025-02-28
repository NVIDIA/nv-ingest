# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import traceback

import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from morpheus.config import Config
from nv_ingest.schemas.audio_extractor_schema import AudioExtractorSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage

from nv_ingest.util.audio.parakeet import call_audio_inference_model
from nv_ingest.util.audio.parakeet import create_audio_inference_client

logger = logging.getLogger(f"morpheus.{__name__}")


def _update_metadata(row: pd.Series, audio_client: Any, trace_info: Dict) -> Dict:
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
    if content_metadata.get("type") != "audio":
        return metadata

    # Modify audio metadata with the result from the inference model
    try:
        audio_result = call_audio_inference_model(audio_client, base64_audio, trace_info=trace_info)
        metadata["audio_metadata"] = {"audio_transcript": audio_result}
    except Exception as e:
        logger.error(f"Unhandled error calling audio inference model: {e}", exc_info=True)
        traceback.print_exc()
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

    _ = task_props

    parakeet_client = create_audio_inference_client(
        validated_config.audio_extraction_config.audio_endpoints,
        auth_token=validated_config.audio_extraction_config.auth_token,
    )

    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    try:
        # Apply the _update_metadata function to each row in the DataFrame
        df["metadata"] = df.apply(_update_metadata, axis=1, args=(parakeet_client, trace_info))

        return df, trace_info

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error occurred while extracting audio data: {e}", exc_info=True)
        raise


def generate_audio_extractor_stage(
    c: Config,
    stage_config: Dict[str, Any],
    task: str = "audio_data_extract",
    task_desc: str = "audio_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform audio data extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the audio content extractor, passed as a dictionary
        validated against the `AudioExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific audio extraction process.
        Default is "audio_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during audio extraction.
        Default is "audio_data_extraction".

    pe_count : int, optional
        The number of process engines to use for audio data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles audio data extraction
        from PDF content.
    """

    validated_config = AudioExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(_transcribe_audio, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c,
        pe_count=pe_count,
        task=task,
        task_desc=task_desc,
        process_fn=_wrapped_process_fn,
        document_type="regex:^(mp3|wav)$",
    )
