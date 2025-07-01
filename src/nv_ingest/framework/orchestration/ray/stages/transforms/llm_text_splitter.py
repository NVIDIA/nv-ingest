# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.schemas.transform.transform_llm_text_splitter_schema import LLMTextSplitterSchema
from nv_ingest_api.internal.transform.llm_split_text import transform_text_split_llm_internal

logger = logging.getLogger(__name__)


def llm_text_splitter_fn(
    control_message: IngestControlMessage, stage_config: LLMTextSplitterSchema
) -> IngestControlMessage:
    """
    UDF for the LLM text splitter stage. This function conforms to the
    pipeline's required signature and orchestrates the call to the core splitting logic.

    Parameters
    ----------
    control_message : IngestControlMessage
        The incoming message containing the payload DataFrame.
    stage_config : LLMTextSplitterSchema
        The stage-level configuration object.

    Returns
    -------
    IngestControlMessage
        The updated message with its payload transformed by the splitting logic.
    """
    df_payload = control_message.payload()
    logger.debug("LLMTextSplitter received payload with %d rows.", len(df_payload))

    # Remove the "split" task to obtain task-specific configuration overrides.
    task_config = remove_task_by_type(control_message, "split")
    logger.debug("Extracted task config: %s", task_config)

    # Transform the DataFrame using the core markdown and LLM splitting logic.
    df_updated = transform_text_split_llm_internal(
        df_transform_ledger=df_payload,
        task_config=task_config,
        transform_config=stage_config,
        execution_trace_log=None,
    )
    logger.info("LLM text splitting complete. Updated payload has %d rows.", len(df_updated))

    # Update the message payload and return.
    control_message.payload(df_updated)
    logger.info("LLMTextSplitter finished processing, returning updated message.")

    return control_message 