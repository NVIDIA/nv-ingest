# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

from pydantic import ConfigDict, BaseModel

logger = logging.getLogger(__name__)


class AssociateNearbyTextSchema(BaseModel):
    n_neighbors: int = 5
    raise_on_failure: bool = False
    model_config = ConfigDict(extra="forbid")
