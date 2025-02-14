# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel

from nv_ingest.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class EmbedExtractionsSchema(BaseModel):
    api_key: str = "api_key"
    batch_size: int = 100
    model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    endpoint_url: str = "http://embedding:8000/v1"
    encoding_format: str = "float"
    httpx_log_level: LogLevel = LogLevel.WARNING
    input_type: str = "passage"
    raise_on_failure: bool = False
    truncate: str = "END"
    model_config = ConfigDict(extra="forbid")
