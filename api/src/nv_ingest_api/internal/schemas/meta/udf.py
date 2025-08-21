# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, ConfigDict


class UDFStageSchema(BaseModel):
    """
    Schema for UDF stage configuration.

    The UDF function string should be provided in the task config. If no UDF function
    is provided and ignore_empty_udf is True, the message is returned unchanged.
    If ignore_empty_udf is False, an error is raised when no UDF function is provided.
    """

    ignore_empty_udf: bool = Field(
        False,
        description="If True, ignore UDF tasks without udf_function and return message unchanged. "
        "If False, raise error.",
    )

    model_config = ConfigDict(extra="forbid")
