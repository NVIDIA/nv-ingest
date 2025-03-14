# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel
from pydantic import StrictBool

logger = logging.getLogger(__name__)


class ImageFilterSchema(BaseModel):
    raise_on_failure: StrictBool = False
    cpu_only: StrictBool = False
    model_config = ConfigDict(extra="forbid")
