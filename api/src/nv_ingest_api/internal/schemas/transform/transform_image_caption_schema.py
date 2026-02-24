# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import ConfigDict, BaseModel, model_validator, field_validator, Field


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str = Field(default="", repr=False)
    endpoint_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    prompt: str = "Caption the content of this image:"
    system_prompt: str = "/no_think"
    model_name: str = "nvidia/nemotron-nano-12b-v2-vl"
    context_text_max_chars: int = 0
    temperature: float = 1.0
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    @classmethod
    def _coerce_api_key_none(cls, v):
        return "" if v is None else v

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_to_defaults(cls, values):
        """Normalize None inputs so validation keeps existing defaults."""
        if not isinstance(values, dict):
            return values

        if values.get("api_key") is None:
            values["api_key"] = ""
        if values.get("prompt") is None:
            values["prompt"] = cls.model_fields["prompt"].default
        if values.get("system_prompt") is None:
            values["system_prompt"] = cls.model_fields["system_prompt"].default
        if values.get("context_text_max_chars") is None:
            values["context_text_max_chars"] = cls.model_fields["context_text_max_chars"].default
        if values.get("temperature") is None:
            values["temperature"] = cls.model_fields["temperature"].default
        return values
