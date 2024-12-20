# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field, BaseModel

from nv_ingest.schemas.message_broker_client_schema import MessageBrokerClientSchema
from typing_extensions import Annotated


class MessageBrokerTaskSourceSchema(BaseModel):
    broker_client: MessageBrokerClientSchema = MessageBrokerClientSchema()

    task_queue: str = "morpheus_task_queue"
    raise_on_failure: bool = False

    progress_engines: Annotated[int, Field(ge=1)] = 6
