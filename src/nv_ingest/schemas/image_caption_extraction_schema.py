# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel


class ImageCaptionExtractionSchema(BaseModel):
    batch_size: int = 8
    caption_classifier_model_name: str = "deberta_large"
    endpoint_url: str = "triton:8001"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
