# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import ConfigDict, BaseModel


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str = "api_key"
    endpoint_url: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    prompt: str = "Caption the content of this image:"
    model_name: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")
