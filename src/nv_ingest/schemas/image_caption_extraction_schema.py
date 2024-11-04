# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel


class ImageCaptionExtractionSchema(BaseModel):
    api_key: str = "api_key"
    endpoint_url: str = "triton:8001"
    prompt: str = "Caption the content of this image:"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
