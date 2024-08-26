# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import BaseModel

from nv_ingest.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class EmbedExtractionsSchema(BaseModel):
    api_key: str
    embedding_nim_endpoint: str
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    encoding_format: str = "float"
    input_type: str = "passage"
    truncate: str = "END"
    batch_size: int = 100
    httpx_log_level: LogLevel = LogLevel.WARNING
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
