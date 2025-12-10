# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel, Field, model_validator, field_validator

from typing import Optional

from nv_ingest_api.util.logging.configuration import LogLevel

logger = logging.getLogger(__name__)


class TextEmbeddingSchema(BaseModel):
    api_key: str = Field(default="", repr=False)
    batch_size: int = Field(default=4)
    embedding_model: str = Field(default="nvidia/llama-3.2-nv-embedqa-1b-v2")
    embedding_nim_endpoint: str = Field(default="http://embedding:8000/v1")
    encoding_format: str = Field(default="float")
    httpx_log_level: LogLevel = Field(default=LogLevel.WARNING)
    input_type: str = Field(default="passage")
    raise_on_failure: bool = Field(default=False)
    truncate: str = Field(default="END")
    text_elements_modality: str = Field(default="text")
    image_elements_modality: str = Field(default="text")
    structured_elements_modality: str = Field(default="text")
    audio_elements_modality: str = Field(default="text")
    custom_content_field: Optional[str] = None
    result_target_field: Optional[str] = None
    dimensions: Optional[int] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    @classmethod
    def _coerce_api_key_none(cls, v):
        return "" if v is None else v

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, values):
        """Convert api_key=None to empty string so validation passes when key is omitted."""
        if isinstance(values, dict) and values.get("api_key") is None:
            values["api_key"] = ""
        return values
