# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import Field, BaseModel

from typing_extensions import Annotated

from nv_ingest_api.internal.schemas.message_brokers.message_broker_client_schema import MessageBrokerClientSchema


class MessageBrokerTaskSourceSchema(BaseModel):
    broker_client: MessageBrokerClientSchema = MessageBrokerClientSchema()

    task_queue: str = "ingest_task_queue"
    raise_on_failure: bool = False

    progress_engines: Annotated[int, Field(ge=1)] = 6
