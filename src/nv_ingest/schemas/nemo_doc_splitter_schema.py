# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Literal
from typing import Optional

from pydantic import Field, BaseModel
from pydantic import validator
from typing_extensions import Annotated


class DocumentSplitterSchema(BaseModel):
    split_by: Literal["word", "sentence", "passage"] = "word"
    split_length: Annotated[int, Field(gt=0)] = 60
    split_overlap: Annotated[int, Field(ge=0)] = 10
    max_character_length: Optional[Annotated[int, Field(gt=0)]] = 450
    sentence_window_size: Optional[Annotated[int, Field(ge=0)]] = 0
    raise_on_failure: bool = False

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("sentence_window_size")
    def check_sentence_window_size(cls, v, values, **kwargs):
        if v is not None and v > 0 and values["split_by"] != "sentence":
            raise ValueError("When using sentence_window_size, split_by must be 'sentence'.")
        return v
