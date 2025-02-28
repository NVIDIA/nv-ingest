# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Dict, Any

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Stage Generation
# ------------------------------------------------------------------------------


def generate_text_embed_extractor_stage(
    c: Any,
    transform_config: Dict[str, Any],
    task: str = "embed",
    task_desc: str = "text_embed_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform text embedding extraction from a pandas DataFrame.

    Parameters
    ----------
    c : Any
        Global configuration object.
    transform_config : Dict[str, Any]
        Configuration parameters for the text embedding extractor, validated against EmbedExtractionsSchema.
    task : str, optional
        The task name for the stage worker function (default is "embed").
    task_desc : str, optional
        A descriptor used for latency tracing and logging (default is "text_embed_extraction").
    pe_count : int, optional
        Number of process engines to use concurrently (default is 1).

    Returns
    -------
    MultiProcessingBaseStage
        A configured stage that processes a pandas DataFrame and returns a tuple of (DataFrame, trace_info dict).
    """
    validated_config = EmbedExtractionsSchema(**transform_config)
    _wrapped_process_fn = functools.partial(
        transform_create_text_embeddings_internal, transform_config=validated_config
    )
    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
