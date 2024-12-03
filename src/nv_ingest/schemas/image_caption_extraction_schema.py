# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str
    endpoint_url: str = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    prompt: str = "Caption the content of this image:"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
