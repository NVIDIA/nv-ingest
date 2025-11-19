# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import ConfigDict, BaseModel, model_validator, field_validator, Field


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str = Field(default="", repr=False)
    endpoint_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    prompt: str = "Caption the content of this image:"
    model_name: str = "nvidia/nemotron-nano-12b-v2-vl"
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")

    @field_validator("api_key", mode="before")
    @classmethod
    def _coerce_api_key_none(cls, v):
        return "" if v is None else v

    @model_validator(mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, values):
        """Allow None for string fields where empty string is acceptable.

        Specifically, convert api_key=None to api_key="" so validation passes
        when no API key is supplied.
        """
        if isinstance(values, dict):
            if values.get("api_key") is None:
                values["api_key"] = ""
        return values
