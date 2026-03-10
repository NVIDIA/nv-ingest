# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, BaseModel, field_validator, ConfigDict

from typing import Optional


class TextSplitterSchema(BaseModel):
    tokenizer: Optional[str] = None
    chunk_size: int = Field(default=1024, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    raise_on_failure: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def check_chunk_overlap(cls, v, values):
        chunk_size = values.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    model_config = ConfigDict(extra="forbid")
