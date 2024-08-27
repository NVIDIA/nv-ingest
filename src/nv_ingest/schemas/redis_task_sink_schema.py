# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel
from pydantic import conint

from nv_ingest.schemas.redis_client_schema import RedisClientSchema


class RedisTaskSinkSchema(BaseModel):
    redis_client: RedisClientSchema = RedisClientSchema()
    raise_on_failure: bool = False

    progress_engines: conint(ge=1) = 6
