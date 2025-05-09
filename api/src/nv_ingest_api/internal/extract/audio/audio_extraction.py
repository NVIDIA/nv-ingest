# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import create_audio_inference_client
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.meta.metadata_schema import MetadataSchema, AudioMetadataSchema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from nv_ingest_api.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


@unified_exception_handler
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
    if (content_metadata.get("type") != ContentTypeEnum.AUDIO) or (base64_audio in (None, "")):
        return metadata

    # Modify audio metadata with the result from the inference model
    audio_result = audio_client.infer(
        base64_audio,
        model_name="parakeet",
        trace_info=trace_info,  # traceable_func arg
        stage_name="audio_extraction",
    )

    row["document_type"] = ContentTypeEnum.AUDIO
    audio_metadata = {"audio_transcript": audio_result}
    metadata["audio_metadata"] = validate_schema(audio_metadata, AudioMetadataSchema).model_dump()
    row["metadata"] = validate_schema(metadata, MetadataSchema).model_dump()

    return metadata


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
        # Apply the _update_metadata function to each row in the DataFrame
        df_extraction_ledger["metadata"] = df_extraction_ledger.apply(
            _update_audio_metadata, axis=1, args=(parakeet_client, execution_trace_log)
        )

        return df_extraction_ledger, execution_trace_log

    except Exception as e:
        logger.exception(f"Error occurred while extracting audio data: {e}", exc_info=True)

        raise
