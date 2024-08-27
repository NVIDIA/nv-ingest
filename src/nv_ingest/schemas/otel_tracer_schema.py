# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel


class OpenTelemetryTracerSchema(BaseModel):
    otel_endpoint: str = "localhost:4317"
    raise_on_failure: bool = False

    class Config:
        extra = "forbid"
