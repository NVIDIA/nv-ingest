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
from typing import Optional

from pydantic import ConfigDict, BaseModel, model_validator

logger = logging.getLogger(__name__)


class ImageStorageModuleSchema(BaseModel):
    structured: bool = True
    images: bool = True
    enable_minio: bool = True
    enable_local_disk: bool = False
    local_output_path: Optional[str] = None
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_storage_targets(cls, values: "ImageStorageModuleSchema") -> "ImageStorageModuleSchema":
        if not values.enable_minio and not values.enable_local_disk:
            raise ValueError("At least one storage backend must be enabled.")

        if values.enable_local_disk and not values.local_output_path:
            raise ValueError("`local_output_path` is required when `enable_local_disk` is True.")

        return values
