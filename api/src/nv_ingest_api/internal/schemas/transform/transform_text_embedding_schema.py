# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel, Field

from nv_ingest_api.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class TextEmbeddingSchema(BaseModel):
    api_key: str = Field(default="api_key")
    batch_size: int = Field(default=4)
    embedding_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2")
    embedding_nim_endpoint: str = Field(default="http://embedding:8000/v1")
    encoding_format: str = Field(default="float")
    httpx_log_level: LogLevel = Field(default=LogLevel.WARNING)
    input_type: str = Field(default="passage")
    raise_on_failure: bool = Field(default=False)
    truncate: str = Field(default="END")

    model_config = ConfigDict(extra="forbid")
