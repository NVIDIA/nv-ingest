# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nv_ingest.schemas.otel_tracer_schema import OpenTelemetryTracerSchema


def test_otel_tracer_schema_defaults():
    schema = OpenTelemetryTracerSchema()
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_otel_tracer_schema_custom_values():
    schema = OpenTelemetryTracerSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True, "Custom value for raise_on_failure should be respected."
