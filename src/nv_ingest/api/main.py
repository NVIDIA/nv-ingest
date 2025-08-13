# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .v1.health import router as HealthApiRouter
from .v1.ingest import router as IngestApiRouter
from .v1.metrics import router as MetricsApiRouter
from .dependency import create_ingest_service

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build once per process; app owns the instance
    app.state.ingest_service = create_ingest_service()
    try:
        yield
    finally:
        svc = getattr(app.state, "ingest_service", None)
        if svc is not None:
            close = getattr(svc, "close", None)
            if callable(close):
                try:
                    maybe = close()
                    import asyncio
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:
                    logger.exception("Error closing ingest service during shutdown")

# nv-ingest FastAPI app declaration
app = FastAPI(
    title="NV-Ingest Microservice",
    description="Service for ingesting heterogenous datatypes",
    version="25.6.2",
    contact={
        "name": "NVIDIA Corporation",
        "url": "https://nvidia.com",
    },
    docs_url="/docs",
)

app.include_router(IngestApiRouter, prefix="/v1")
app.include_router(HealthApiRouter, prefix="/v1/health")
app.include_router(MetricsApiRouter, prefix="/v1")

# Set up the tracer provider and add a processor for exporting traces
resource = Resource(attributes={"service.name": "nv-ingest"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel-collector:4317")
exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
