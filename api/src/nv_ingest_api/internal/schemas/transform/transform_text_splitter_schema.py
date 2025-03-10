# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field, BaseModel, field_validator

from typing import Optional

from typing_extensions import Annotated


class TextSplitterSchema(BaseModel):
    tokenizer: Optional[str] = None
    chunk_size: Annotated[int, Field(gt=0)] = 1024
    chunk_overlap: Annotated[int, Field(ge=0)] = 150
    raise_on_failure: bool = False

    @field_validator("chunk_overlap")
    def check_chunk_overlap(cls, v, values, **kwargs):
        if v is not None and "chunk_size" in values.data and v >= values.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
