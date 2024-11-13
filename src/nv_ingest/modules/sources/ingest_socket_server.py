# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import traceback
from datetime import datetime
from functools import partial
from typing import Dict
import copy, json

import cudf
import mrc
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from opentelemetry.trace.span import format_trace_id

from nv_ingest.schemas import validate_ingest_job
#from nv_ingest.schemas.ingest_server_schema import IngestServerSchema
from nv_ingest.util.message_brokers.simple_message_broker import SimpleMessageBroker
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing.logging import annotate_cm

logger = logging.getLogger(__name__)

MODULE_NAME = "socket_task_source"
MODULE_NAMESPACE = "nv_ingest"
SocketTaskSourceLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def fetch_and_process_messages(ingest_server: SimpleMessageBroker, validated_config: IngestServerSchema):
    """Fetch jobs from the IngestServer socket and process them."""

    while True:
        try:
            job = ingest_server.receive_job()  # Assume receive_job handles connection
            ts_fetched = datetime.now()
            yield process_message(job, ts_fetched)  # process_message remains unchanged
        except Exception as err:
            logger.error(
                f"Irrecoverable error occurred during message processing, likely malformed JOB structure: {err}"
            )
            traceback.print_exc()


def process_message(job: Dict, ts_fetched: datetime) -> ControlMessage:
    """
    Process incoming job data from the IngestServer and return as ControlMessage.
    """

    if logger.isEnabledFor(logging.DEBUG):
        no_payload = copy.deepcopy(job)
        no_payload["job_payload"]["content"] = ["[...]"]  # Redact the payload for logging
        logger.debug("Job: %s", json.dumps(no_payload, indent=2))

    validate_ingest_job(job)
    control_message = ControlMessage()

    try:
        ts_entry = datetime.now()

        job_id = job.pop("job_id")
        job_payload = job.pop("job_payload", {})
        job_tasks = job.pop("tasks", [])

        tracing_options = job.pop("tracing_options", {})
        do_trace_tagging = tracing_options.get("trace", False)
        ts_send = tracing_options.get("ts_send", None)
        if ts_send is not None:
            ts_send = datetime.fromtimestamp(ts_send / 1e9)
        trace_id = tracing_options.get("trace_id", None)

        response_channel = f"response_{job_id}"

        df = cudf.DataFrame(job_payload)
        message_meta = MessageMeta(df=df)

        control_message.payload(message_meta)
        annotate_cm(control_message, message="Created")
        control_message.set_metadata("response_channel", response_channel)
        control_message.set_metadata("job_id", job_id)

        for task in job_tasks:
            control_message.add_task(task["type"], task["task_properties"])

        if do_trace_tagging:
            ts_exit = datetime.now()
            control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
            control_message.set_timestamp(f"trace::entry::{MODULE_NAME}", ts_entry)
            control_message.set_timestamp(f"trace::exit::{MODULE_NAME}", ts_exit)

            if ts_send is not None:
                control_message.set_timestamp("trace::entry::socket_source_network_in", ts_send)
                control_message.set_timestamp("trace::exit::socket_source_network_in", ts_fetched)

            if trace_id is not None:
                if isinstance(trace_id, int):
                    trace_id = format_trace_id(trace_id)
                control_message.set_metadata("trace_id", trace_id)

            control_message.set_timestamp("latency::ts_send", datetime.now())
    except Exception as e:
        if "job_id" in job:
            job_id = job["job_id"]
            response_channel = f"response_{job_id}"
            control_message.set_metadata("job_id", job_id)
            control_message.set_metadata("response_channel", response_channel)
            control_message.set_metadata("cm_failed", True)
            annotate_cm(control_message, message="Failed to process job submission", error=str(e))
        else:
            raise

    return control_message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _socket_task_source(builder: mrc.Builder):
    """
    A module for receiving messages from a socket connection, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """

    validated_config = fetch_and_validate_module_config(builder, IngestServerSchema)

    ingest_server = SimpleMessageBroker(
        host=validated_config.ingest_server.hostname,
        port=validated_config.ingest_server.port,
        max_retries=validated_config.ingest_server.max_retries,
        connection_timeout=validated_config.ingest_server.connection_timeout,
    )

    _fetch_and_process_messages = partial(
        fetch_and_process_messages,
        ingest_server=ingest_server,
        validated_config=validated_config,
    )

    node = builder.make_source("fetch_messages_socket", _fetch_and_process_messages)
    node.launch_options.engines_per_pe = validated_config.progress_engines

    builder.register_module_output("output", node)
