# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nv_ingest_api.internal.extract.pptx.pptx_extractor import extract_primitives_from_pptx_internal
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema

logger = logging.getLogger(__name__)


def pptx_extractor_udf(control_message: IngestControlMessage, config: PPTXExtractorSchema) -> IngestControlMessage:
    """
    Processes a control message by extracting content from PPTX documents.

    Parameters
    ----------
    control_message : IngestControlMessage
        The message containing a DataFrame payload with PPTX document data.
    config : PPTXExtractorSchema
            The validated configuration object for PPTX extraction.

    Returns
    -------
    IngestControlMessage
        The updated message with extracted PPTX content.
    """

    # Extract the DataFrame payload.
    df_payload = control_message.payload()

    # Remove the "pptx-extract" task from the message to obtain task-specific configuration.
    task_config = remove_task_by_type(control_message, "extract")
    logger.debug("Extracted task config: %s", task_config)

    new_df, extraction_info = extract_primitives_from_pptx_internal(
        df_extraction_ledger=df_payload,
        task_config=task_config,
        extraction_config=config,
        execution_trace_log=None,  # Assuming None is appropriate here as in DOCX example
    )

    # Update the message payload with the extracted PPTX content DataFrame.
    control_message.payload(new_df)
    control_message.set_metadata("pptx_extraction_info", extraction_info)  # <-- Changed metadata key

    return control_message
