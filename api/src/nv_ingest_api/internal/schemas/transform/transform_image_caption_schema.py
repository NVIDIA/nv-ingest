# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import ConfigDict, BaseModel


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str = "api_key"
    endpoint_url: str = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    prompt: str = "Caption the content of this image:"
    image_caption_model_name: str = "meta/llama-3.2-11b-vision-instruct"
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")
