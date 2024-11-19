# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel
from pydantic import conint

from nv_ingest.schemas.message_broker_client_schema import MessageBrokerClientSchema


class MessageBrokerTaskSourceSchema(BaseModel):
    client_specification : MessageBrokerClientSchema = MessageBrokerClientSchema()

    task_queue: str = "morpheus_task_queue"

    progress_engines: conint(ge=1) = 6
