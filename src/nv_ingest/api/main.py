# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI

from .v1.ingest import router as IngestApiRouter

# Set up the tracer provider and add a processor for exporting traces
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

exporter = OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# nv-ingest FastAPI app declaration
app = FastAPI()

app.include_router(IngestApiRouter)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def add_trace_id_header(request, call_next):
    with tracer.start_as_current_span("uvicorn-endpoint"):
        response = await call_next(request)

        # Inject the current x-trace-id into the HTTP headers response
        span = trace.get_current_span()
        if span:
            trace_id = format(span.get_span_context().trace_id, '032x')
            response.headers["x-trace-id"] = trace_id

        return response
