# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field, BaseModel

from typing_extensions import Annotated


class TextSplitterSchema(BaseModel):
    tokenizer: str = "intfloat/e5-large-unsupervised"
    chunk_size: Annotated[int, Field(gt=0)] = 300
    chunk_overlap: Annotated[int, Field(ge=0)] = 0
    raise_on_failure: bool = False
