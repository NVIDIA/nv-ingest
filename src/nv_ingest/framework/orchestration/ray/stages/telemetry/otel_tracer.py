# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Optional

import ray
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

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.schemas.framework_otel_tracer_schema import OpenTelemetryTracerSchema
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except

from nv_ingest_api.internal.primitives.tracing.logging import TaskResultStatus
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook


@ray.remote
class OpenTelemetryTracerStage(RayActorStage):
    """
    A Ray actor stage that collects and exports traces to OpenTelemetry.

    This stage uses OpenTelemetry to trace the execution of tasks within the system.
    It creates spans for tasks and exports them to a configured OpenTelemetry endpoint.
    """

    def __init__(self, config: OpenTelemetryTracerSchema, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)

        # self._logger.info(f"[Telemetry] Initializing OpenTelemetry tracer stage with config: {config}")

        self.validated_config: OpenTelemetryTracerSchema = config
        self.resource = Resource(attributes={"service.name": "nv-ingest"})
        self.otlp_exporter = OTLPSpanExporter(endpoint=self.validated_config.otel_endpoint, insecure=True)
        self.span_processor = BatchSpanProcessor(self.otlp_exporter)

        trace.set_tracer_provider(TracerProvider(resource=self.resource))
        trace.get_tracer_provider().add_span_processor(self.span_processor)

        self.tracer = trace.get_tracer(__name__)

    def collect_timestamps(self, message):
        job_id = message.get_metadata("job_id")
        if isinstance(job_id, str) and len(job_id) == 36:
            trace_id = uuid_to_trace_id(job_id)
        elif isinstance(job_id, str):
            trace_id = int(job_id, 16)
        else:
            trace_id = RandomIdGenerator().generate_trace_id()

        span_id = RandomIdGenerator().generate_span_id()
        timestamps = extract_timestamps_from_message(message)

        flattened = [x for t in timestamps.values() for x in t]
        if not flattened:
            self._logger.debug("No timestamps found for message; skipping tracing.")
            return

        start_time = min(flattened)
        end_time = max(flattened)

        self._logger.debug(f"[Telemetry] trace_id: {trace_id}, span_id: {span_id}")

        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=TraceFlags(0x01),
        )
        parent_ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
        parent_span = self.tracer.start_span(str(job_id), context=parent_ctx, start_time=start_time)

        event_count = create_span_with_timestamps(self.tracer, parent_span, message, self._logger)

        if message.has_metadata("cm_failed") and message.get_metadata("cm_failed"):
            parent_span.set_status(Status(StatusCode.ERROR))
        else:
            parent_span.set_status(Status(StatusCode.OK))

        try:
            parent_span.add_event("start", timestamp=start_time)
            parent_span.add_event("end", timestamp=end_time)
        finally:
            parent_span.end(end_time=end_time)

        self._logger.debug(f"[Telemetry] Exported spans for message {job_id} with {event_count} total events.")

    @nv_ingest_node_failure_try_except()
    @udf_intercept_hook()
    def on_data(self, control_message: IngestControlMessage) -> Optional[Any]:
        try:
            do_trace_tagging = bool(control_message.get_metadata("config::add_trace_tagging"))

            if not do_trace_tagging:
                self._logger.debug("Skipping OpenTelemetry tracing, do_trace_tagging is False.")
                return control_message

            self._logger.debug("Sending telemetry data to OpenTelemetry")

            self.collect_timestamps(control_message)

            return control_message
        except Exception as e:
            self._logger.warning(f"Error in OpenTelemetry tracer stage: {e}")
            raise e


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
        if ts_entry is None:
            continue

        ts_exit = (
            message.get_timestamp(exit_key) or datetime.now()
        )  # When a job fails, it may not have exit time. Default to current time.
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


def create_span_with_timestamps(tracer, parent_span, message, logger) -> int:
    timestamps = extract_timestamps_from_message(message)
    task_results = extract_annotated_task_results(message)

    ctx_store = {}
    event_counter = 0
    child_ctx = trace.set_span_in_context(parent_span)

    for task_name, (ts_entry, ts_exit) in sorted(timestamps.items(), key=lambda x: x[1]):
        main_task, *subtask = task_name.split("::", 1)
        subtask = "::".join(subtask)

        if not subtask:
            span = tracer.start_span(main_task, context=child_ctx, start_time=ts_entry)
        else:
            # Check if parent context exists, otherwise create standalone span with warning
            if main_task in ctx_store:
                subtask_ctx = trace.set_span_in_context(ctx_store[main_task][0])
                span = tracer.start_span(subtask, context=subtask_ctx, start_time=ts_entry)
            else:
                logger.warning(
                    f"Missing parent context for subtask '{subtask}'"
                    f" (expected parent: '{main_task}'). Creating standalone span."
                )
                span = tracer.start_span(f"{main_task}::{subtask}", context=child_ctx, start_time=ts_entry)

        span.add_event("entry", timestamp=ts_entry)
        span.add_event("exit", timestamp=ts_exit)
        event_counter += 2

        if task_name in task_results:
            task_result = task_results[main_task]
            if task_result == TaskResultStatus.SUCCESS.value:
                span.set_status(Status(StatusCode.OK))
            if task_result == TaskResultStatus.FAILURE.value:
                span.set_status(Status(StatusCode.ERROR))

        ctx_store[task_name] = (span, ts_exit)

    for _, (span, ts_exit) in ctx_store.items():
        span.end(end_time=ts_exit)

    return event_counter


def uuid_to_trace_id(uuid_str: str) -> int:
    """Convert a UUID-like string to an integer OpenTelemetry trace ID."""
    if not isinstance(uuid_str, str) or len(uuid_str) != 36:
        raise ValueError("UUID must be a 36-character string with hyphens")

    return int(uuid_str.replace("-", ""), 16)
