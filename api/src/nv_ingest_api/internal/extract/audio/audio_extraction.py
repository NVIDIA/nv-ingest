# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
import functools
import uuid
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import base64
from pathlib import Path

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import create_audio_inference_client
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.meta.metadata_schema import MetadataSchema, AudioMetadataSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from nv_ingest_api.util.schema.schema_validator import validate_schema
from nv_ingest_api.interface.utility import read_file_as_base64

logger = logging.getLogger(__name__)


@unified_exception_handler
def _extract_from_audio(row: pd.Series, audio_client: Any, trace_info: Dict, segment_audio: bool = False) -> Dict:
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
    try:
        base64_file_path = base64_audio
        if not base64_file_path:
            return [row.to_list()]
        base64_file_path = base64.b64decode(base64_file_path).decode("utf-8")
        if not base64_file_path:
            return [row.to_list()]
        if Path(base64_file_path).exists():
            base64_audio = read_file_as_base64(base64_file_path)
    except (UnicodeDecodeError, base64.binascii.Error):
        pass
    content_metadata = metadata.get("content_metadata", {})

    # Only extract transcript if content type is audio
    if (content_metadata.get("type") != ContentTypeEnum.AUDIO) or (base64_audio in (None, "")):
        return [row.to_list()]

    logger.error(f"Removing file {base64_file_path}")
    Path(base64_file_path).unlink(missing_ok=True)

    # Get the result from the inference model
    segments, transcript = audio_client.infer(
        base64_audio,
        model_name="parakeet",
        trace_info=trace_info,  # traceable_func arg
        stage_name="audio_extraction",
    )

    extracted_data = []
    if segment_audio:
        for segment in segments:
            segment_metadata = metadata.copy()
            audio_metadata = {"audio_transcript": segment["text"]}
            segment_metadata["audio_metadata"] = validate_schema(audio_metadata, AudioMetadataSchema).model_dump()
            segment_metadata["content_metadata"]["start_time"] = segment["start"]
            segment_metadata["content_metadata"]["end_time"] = segment["end"]

            extracted_data.append(
                [
                    ContentTypeEnum.AUDIO,
                    validate_schema(segment_metadata, MetadataSchema).model_dump(),
                    str(uuid.uuid4()),
                ]
            )
    else:
        audio_metadata = {"audio_transcript": transcript}
        metadata["audio_metadata"] = validate_schema(audio_metadata, AudioMetadataSchema).model_dump()
        extracted_data.append(
            [ContentTypeEnum.AUDIO, validate_schema(metadata, MetadataSchema).model_dump(), str(uuid.uuid4())]
        )

    return extracted_data


def extract_text_from_audio_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    extraction_config: AudioExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts audio data from a DataFrame.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which audio data is to be extracted.

    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.

    extraction_config : Any
        The validated configuration object for audio extraction.

    execution_trace_log : Optional[Dict], optional
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
    logger.debug(f"Entering audio extraction stage with {len(df_extraction_ledger)} rows.")

    extract_params = task_config.get("params", {}).get("extract_audio_params", {})
    audio_extraction_config = extraction_config.audio_extraction_config

    grpc_endpoint = extract_params.get("grpc_endpoint") or audio_extraction_config.audio_endpoints[0]
    http_endpoint = extract_params.get("http_endpoint") or audio_extraction_config.audio_endpoints[1]
    infer_protocol = extract_params.get("infer_protocol") or audio_extraction_config.audio_infer_protocol
    auth_token = extract_params.get("auth_token") or audio_extraction_config.auth_token
    function_id = extract_params.get("function_id") or audio_extraction_config.function_id
    use_ssl = extract_params.get("use_ssl") or audio_extraction_config.use_ssl
    ssl_cert = extract_params.get("ssl_cert") or audio_extraction_config.ssl_cert
    segment_audio = extract_params.get("segment_audio") or audio_extraction_config.segment_audio

    parakeet_client = create_audio_inference_client(
        (grpc_endpoint, http_endpoint),
        infer_protocol=infer_protocol,
        auth_token=auth_token,
        function_id=function_id,
        use_ssl=use_ssl,
        ssl_cert=ssl_cert,
    )

    if execution_trace_log is None:
        execution_trace_log = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    try:
        # Create a partial function to extract using the provided configurations.
        _extract_from_audio_partial = functools.partial(
            _extract_from_audio,
            audio_client=parakeet_client,
            trace_info=execution_trace_log,
            segment_audio=segment_audio,
        )

        # Apply the _extract_from_audio_partial function to each row in the DataFrame
        extraction_series = df_extraction_ledger.apply(_extract_from_audio_partial, axis=1)

        # Explode the results if the extraction returns lists.
        extraction_series = extraction_series.explode().dropna()

        # Convert the extracted results into a DataFrame.
        if not extraction_series.empty:
            extracted_df = pd.DataFrame(extraction_series.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, execution_trace_log

    except Exception as e:
        logger.exception(f"Error occurred while extracting audio data: {e}", exc_info=True)

        raise
