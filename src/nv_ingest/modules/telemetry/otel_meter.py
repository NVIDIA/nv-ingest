# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import re
import traceback
from datetime import datetime

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from nv_ingest.schemas.otel_meter_schema import OpenTelemetryMeterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.redis import RedisClient
from nv_ingest.util.telemetry.global_stats import GlobalStats

logger = logging.getLogger(__name__)

MODULE_NAME = "opentelemetry_meter"
MODULE_NAMESPACE = "nv_ingest"

OpenTelemetryMeterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def sanitize_name(name):
    # Replace spaces with underscores and ensure only ASCII characters
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name.replace(" ", "_"))
    # Truncate the name to a maximum of 63 characters
    return sanitized[:63]


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _metrics_aggregation(builder: mrc.Builder) -> None:
    """
    Module for collecting and exporting job statistics to OpenTelemetry

    Parameters
    ----------
    builder : mrc.Builder
        The module configuration builder.

    Returns
    -------
    None
    """
    validated_config = fetch_and_validate_module_config(builder, OpenTelemetryMeterSchema)
    stats = GlobalStats.get_instance()

    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is always 0 for simplicity; make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    resource = Resource(attributes={"service.name": "nv-ingest"})

    reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=validated_config.otel_endpoint, insecure=True))
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

    set_global_textmap(TraceContextTextMapPropagator())

    meter = metrics.get_meter(__name__)

    gauges = {
        "inflight_jobs_total": meter.create_gauge("inflight_jobs_total"),
        "completed_jobs_total": meter.create_gauge("completed_jobs_total"),
        "failed_jobs_total": meter.create_gauge("failed_jobs_total"),
        "source_to_sink_mean": meter.create_gauge("source_to_sink_mean"),
        "source_to_sink_median": meter.create_gauge("source_to_sink_median"),
        "outstanding_job_responses_total": meter.create_gauge("outstanding_job_responses_total"),
        "response_wait_time_mean": meter.create_gauge("response_wait_time_mean"),
        "response_wait_time_median": meter.create_gauge("response_wait_time_median"),
    }

    response_channels_store = {}

    def update_job_stats():
        submitted_jobs = stats.get_stat("submitted_jobs")
        completed_jobs = stats.get_stat("completed_jobs")
        failed_jobs = stats.get_stat("failed_jobs")
        inflight_jobs = submitted_jobs - completed_jobs - failed_jobs
        gauges["inflight_jobs_total"].set(inflight_jobs)
        gauges["completed_jobs_total"].set(completed_jobs)
        gauges["failed_jobs_total"].set(failed_jobs)

    def update_job_latency(message):
        for key, val in message.filter_timestamp("trace::exit::").items():
            exit_key = key
            entry_key = exit_key.replace("trace::exit::", "trace::entry::")
            ts_exit = val
            ts_entry = message.get_timestamp(entry_key)
            job_name = key.replace("trace::exit::", "")

            # Sanitize job name
            sanitized_job_name = sanitize_name(job_name)

            latency_ms = (ts_exit - ts_entry).total_seconds() * 1e3

            stats.append_job_stat(sanitized_job_name, latency_ms)
            mean = stats.get_job_stat(sanitized_job_name, "mean")
            median = stats.get_job_stat(sanitized_job_name, "median")

            mean_gauge_name = f"{sanitized_job_name}_mean"
            median_gauge_name = f"{sanitized_job_name}_median"

            if mean_gauge_name not in gauges:
                gauges[mean_gauge_name] = meter.create_gauge(mean_gauge_name)
            if median_gauge_name not in gauges:
                gauges[median_gauge_name] = meter.create_gauge(median_gauge_name)

            gauges[mean_gauge_name].set(mean)
            gauges[median_gauge_name].set(median)

    def update_e2e_latency(message):
        created_ts = pushed_ts = None
        for key, val in message.filter_timestamp("annotation::").items():
            annotation_message = key.replace("annotation::", "")
            if annotation_message == "Created":
                created_ts = message.get_timestamp(key)
            if annotation_message == "Pushed":
                pushed_ts = message.get_timestamp(key)

        if created_ts and pushed_ts:
            latency_ms = (pushed_ts - created_ts).total_seconds() * 1e3
            stats.append_job_stat("source_to_sink", latency_ms)
            mean = stats.get_job_stat("source_to_sink", "mean")
            median = stats.get_job_stat("source_to_sink", "median")
            gauges["source_to_sink_mean"].set(mean)
            gauges["source_to_sink_median"].set(median)

    def update_response_stats(message):
        response_channel = message.get_metadata("response_channel")
        response_channels_store[response_channel] = message.get_timestamp("annotation::Pushed")

        curr_response_channels = set(
            k for k in redis_client.get_client().keys() if k.decode("utf-8").startswith("response")
        )
        gauges["outstanding_job_responses_total"].set(len(curr_response_channels))

        to_remove = []
        for key, pushed_ts in response_channels_store.items():
            if key in curr_response_channels:
                continue
            to_remove.append(key)
            wait_time_ms = (datetime.now() - pushed_ts).total_seconds() * 1e3  # best effort
            stats.append_job_stat("response_wait_time", wait_time_ms)
            mean = stats.get_job_stat("response_wait_time", "mean")
            median = stats.get_job_stat("response_wait_time", "median")

        gauges["response_wait_time_mean"].set(mean)
        gauges["response_wait_time_median"].set(median)

        for key in to_remove:
            del response_channels_store[key]

    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
        skip_processing_if_failed=False,
    )
    def aggregate_metrics(message: ControlMessage) -> ControlMessage:
        try:
            do_trace_tagging = message.get_metadata("config::add_trace_tagging") is True
            if not do_trace_tagging:
                return message

            logger.debug("Performing statistics aggregation.")

            update_job_stats()
            update_job_latency(message)
            update_e2e_latency(message)
            update_response_stats(message)

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to perform statistics aggregation: {e}")

    aggregate_node = builder.make_node("opentelemetry_meter", ops.map(aggregate_metrics))

    # Register the input and output of the module
    builder.register_module_input("input", aggregate_node)
    builder.register_module_output("output", aggregate_node)
