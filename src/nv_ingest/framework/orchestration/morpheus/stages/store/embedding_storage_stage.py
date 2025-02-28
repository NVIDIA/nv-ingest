# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging

from morpheus.config import Config

from nv_ingest.schemas.embedding_storage_schema import EmbeddingStorageSchema
from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.store.embed_text_upload import store_embeddings_internal

logger = logging.getLogger(__name__)


def generate_embedding_storage_stage(
    c: Config,
    store_config: dict,
    task: str = "store_embedding",
    task_desc: str = "Store_embeddings_minio",
    pe_count: int = 24,
):
    """
    Helper function to generate a multiprocessing stage to perform pdf content extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object
    embedding_storage_config : dict
        Configuration parameters for embedding storage.
    store_config : dict
        Configuration parameters for the embedding storage, passed as a dictionary
        validated against the `EmbeddingStorageModuleSchema`.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        Integer for how many process engines to use for pdf content extraction.

    Returns
    -------
    MultiProcessingBaseStage
        A Morpheus stage with applied worker function.
    """

    try:
        validated_config = EmbeddingStorageSchema(**store_config)

        _wrapped_process_fn = functools.partial(store_embeddings_internal, store_config=validated_config)

        return MultiProcessingBaseStage(
            c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
        )

    except Exception as e:
        err_msg = f"generate_embedding_storage_stage: Error generating embedding storage stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
