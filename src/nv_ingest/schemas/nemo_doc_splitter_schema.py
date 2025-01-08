# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel
from pydantic import conint


class DocumentSplitterSchema(BaseModel):
    tokenizer: str = "intfloat/e5-large-unsupervised"
    chunk_size: conint(gt=0) = 300
    chunk_overlap: conint(ge=0) = 0
    raise_on_failure: bool = False
