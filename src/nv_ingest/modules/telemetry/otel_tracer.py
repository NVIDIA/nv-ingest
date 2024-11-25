# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import traceback

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.trace import SpanContext
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from opentelemetry.trace import TraceFlags

from nv_ingest.schemas.otel_tracer_schema import OpenTelemetryTracerSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing.logging import TaskResultStatus

logger = logging.getLogger(__name__)

MODULE_NAME = "opentelemetry_tracer"
MODULE_NAMESPACE = "nv_ingest"

OpenTelemetryTracerLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _trace(builder: mrc.Builder) -> None:
    """
    Module for collecting and exporting traces to OpenTelemetry.

    Parameters
    ----------
    builder : mrc.Builder
        The module configuration builder.

    Returns
    -------
    None
    """
    validated_config = fetch_and_validate_module_config(builder, OpenTelemetryTracerSchema)

    resource = Resource(attributes={"service.name": "nv-ingest"})

    trace.set_tracer_provider(TracerProvider(resource=resource))

    otlp_exporter = OTLPSpanExporter(endpoint=validated_config.otel_endpoint, insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    tracer = trace.get_tracer(__name__)

    def collect_timestamps(message):
        job_id = message.get_metadata("job_id")

        trace_id = message.get_metadata("trace_id")
        if trace_id is None:
            trace_id = RandomIdGenerator().generate_trace_id()
        elif isinstance(trace_id, str):
            trace_id = int(trace_id, 16)
        span_id = RandomIdGenerator().generate_span_id()

        timestamps = extract_timestamps_from_message(message)

        flattened = [x for t in timestamps.values() for x in t]
        start_time = min(flattened)
        end_time = max(flattened)

        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        parent_ctx = trace.set_span_in_context(span_context)
        parent_span = tracer.start_span(job_id, context=parent_ctx, start_time=start_time)

        create_span_with_timestamps(tracer, parent_span, message)

        if message.has_metadata("cm_failed") and message.get_metadata("cm_failed"):
            parent_span.set_status(Status(StatusCode.ERROR))
        else:
            parent_span.set_status(Status(StatusCode.OK))

        try:
            parent_span.add_event("start", timestamp=start_time)
            parent_span.add_event("end", timestamp=end_time)
        finally:
            parent_span.end(end_time=end_time)

    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
        skip_processing_if_failed=False,
    )
    def on_next(message: ControlMessage) -> ControlMessage:
        try:
            do_trace_tagging = message.get_metadata("config::add_trace_tagging") is True
            if not do_trace_tagging:
                return message

            logger.debug("Sending traces to OpenTelemetry collector.")

            collect_timestamps(message)

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to perform statistics aggregation: {e}")

    aggregate_node = builder.make_node("stats_aggregation", ops.map(on_next))

    builder.register_module_input("input", aggregate_node)
    builder.register_module_output("output", aggregate_node)


def extract_timestamps_from_message(message):
    timestamps = {}
    dedup_counter = {}

    for key, val in message.filter_timestamp("trace::exit::").items():
        exit_key = key
        entry_key = exit_key.replace("trace::exit::", "trace::entry::")

        task_name = key.replace("trace::exit::", "")
        if task_name in dedup_counter:
            dedup_counter[task_name] += 1
            task_name = task_name + "_" + str(dedup_counter[task_name])
        else:
            dedup_counter[task_name] = 0

        ts_entry = message.get_timestamp(entry_key)
        ts_exit = message.get_timestamp(exit_key)
        ts_entry_ns = int(ts_entry.timestamp() * 1e9)
        ts_exit_ns = int(ts_exit.timestamp() * 1e9)

        timestamps[task_name] = (ts_entry_ns, ts_exit_ns)

    return timestamps


def extract_annotated_task_results(message):
    task_results = {}
    for key in message.list_metadata():
        if not key.startswith("annotation::"):
            continue
        task = message.get_metadata(key)
        if not (("task_id" in task) and ("task_result" in task)):
            continue
        task_id = task["task_id"]
        task_result = task["task_result"]
        task_results[task_id] = task_result

    return task_results


def create_span_with_timestamps(tracer, parent_span, message):
    timestamps = extract_timestamps_from_message(message)
    task_results = extract_annotated_task_results(message)

    ctx_store = {}
    child_ctx = trace.set_span_in_context(parent_span)
    for task_name, (ts_entry, ts_exit) in sorted(timestamps.items(), key=lambda x: x[1]):
        main_task, *subtask = task_name.split("::", 1)
        subtask = "::".join(subtask)

        if not subtask:
            span = tracer.start_span(main_task, context=child_ctx, start_time=ts_entry)
        else:
            subtask_ctx = trace.set_span_in_context(ctx_store[main_task][0])
            span = tracer.start_span(subtask, context=subtask_ctx, start_time=ts_entry)

        span.add_event("entry", timestamp=ts_entry)
        span.add_event("exit", timestamp=ts_exit)

        # Set success/failure status.
        if task_name in task_results:
            task_result = task_results[main_task]
            if task_result == TaskResultStatus.SUCCESS.value:
                span.set_status(Status(StatusCode.OK))
            if task_result == TaskResultStatus.FAILURE.value:
                span.set_status(Status(StatusCode.ERROR))

        # Add timestamps.
        span.add_event("entry", timestamp=ts_entry)
        span.add_event("exit", timestamp=ts_exit)

        # Cache span and exit time.
        # Spans are used for looking up the main task's span when creating a subtask's span.
        # Exit timestamps are used for closing each span at the very end.
        ctx_store[task_name] = (span, ts_exit)

    for _, (span, ts_exit) in ctx_store.items():
        span.end(end_time=ts_exit)
