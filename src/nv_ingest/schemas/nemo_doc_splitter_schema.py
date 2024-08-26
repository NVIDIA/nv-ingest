# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import conint
from pydantic import validator


class DocumentSplitterSchema(BaseModel):
    split_by: Literal["word", "sentence", "passage"] = "word"
    split_length: conint(gt=0) = 60
    split_overlap: conint(ge=0) = 10
    max_character_length: Optional[conint(gt=0)] = 450
    sentence_window_size: Optional[conint(ge=0)] = 0
    raise_on_failure: bool = False

    @validator("sentence_window_size")
    def check_sentence_window_size(cls, v, values, **kwargs):
        if v is not None and v > 0 and values["split_by"] != "sentence":
            raise ValueError("When using sentence_window_size, split_by must be 'sentence'.")
        return v
