# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel
from pydantic import conint

from nv_ingest.schemas.message_broker_client_schema import MessageBrokerClientSchema


class MessageBrokerTaskSinkSchema(BaseModel):
    broker_client: MessageBrokerClientSchema = MessageBrokerClientSchema()

    raise_on_failure: bool = False

    progress_engines: conint(ge=1) = 6
