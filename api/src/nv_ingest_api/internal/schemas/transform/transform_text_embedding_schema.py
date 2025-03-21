# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel

from nv_ingest_api.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class TextEmbeddingSchema(BaseModel):
    api_key: str = "api_key"
    batch_size: int = 4
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embedding_nim_endpoint: str = "http://embedding:8000/v1"
    encoding_format: str = "float"
    httpx_log_level: LogLevel = LogLevel.WARNING
    input_type: str = "passage"
    raise_on_failure: bool = False
    truncate: str = "END"
    model_config = ConfigDict(extra="forbid")
