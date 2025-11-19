# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Optional, Dict, Any

from pydantic import ConfigDict, BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


_DEFAULT_STORAGE_URI = os.environ.get("IMAGE_STORAGE_URI", "s3://nv-ingest/artifacts/store/images")


class ImageStorageModuleSchema(BaseModel):
    structured: bool = True
    images: bool = True
    storage_uri: str = Field(default_factory=lambda: _DEFAULT_STORAGE_URI)
    storage_options: Dict[str, Any] = Field(default_factory=dict)
    public_base_url: Optional[str] = None
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")

    @field_validator("storage_uri")
    @classmethod
    def validate_storage_uri(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("`storage_uri` must be provided.")
        return value
