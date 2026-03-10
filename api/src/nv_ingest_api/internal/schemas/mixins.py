# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared mixins for Pydantic schemas.
"""

from typing import Any
from pydantic import BaseModel, field_validator


class LowercaseProtocolMixin(BaseModel):
    """
    Mixin that automatically lowercases any field ending with '_infer_protocol'.

    This ensures case-insensitive handling of protocol values (e.g., "HTTP" -> "http").
    Apply this mixin to any schema that has protocol fields to normalize user input.

    Examples
    --------
    >>> class MyConfigSchema(LowercaseProtocolMixin):
    ...     yolox_infer_protocol: str = ""
    ...     ocr_infer_protocol: str = ""
    >>>
    >>> config = MyConfigSchema(yolox_infer_protocol="GRPC", ocr_infer_protocol="HTTP")
    >>> config.yolox_infer_protocol
    'grpc'
    >>> config.ocr_infer_protocol
    'http'
    """

    @field_validator("*", mode="before")
    @classmethod
    def _lowercase_protocol_fields(cls, v: Any, info):
        """Lowercase any field ending with '_infer_protocol'."""
        if info.field_name.endswith("_infer_protocol") and v is not None:
            return str(v).strip().lower()
        return v
