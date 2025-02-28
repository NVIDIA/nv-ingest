# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import pandas as pd

from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler


@unified_exception_handler
def transform_create_text_embeddings(
    df_transform_ledger: pd.DataFrame,
    api_key: str,
    embedding_model: Optional[str] = None,
    embedding_nim_endpoint: Optional[str] = None,
    encoding_format: Optional[str] = None,
    input_type: Optional[str] = None,
    truncate: Optional[str] = None,
):
    """
    Creates text embeddings using the provided configuration.
    Parameters provided as None will use the default values from EmbedExtractionsSchema.
    """
    task_config = {}

    # Build configuration parameters only if provided; defaults come from EmbedExtractionsSchema.
    config_kwargs = {
        "embedding_model": embedding_model,
        "embedding_nim_endpoint": embedding_nim_endpoint,
        "encoding_format": encoding_format,
        "input_type": input_type,
        "truncate": truncate,
    }
    # Remove any keys with a None value.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config_kwargs["api_key"] = api_key

    transform_config = EmbedExtractionsSchema(**config_kwargs)

    result, _ = transform_create_text_embeddings_internal(
        df_transform_ledger=df_transform_ledger,
        task_config=task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )

    return result


def split_text():
    pass
