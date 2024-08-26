# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import traceback
from datetime import datetime
from functools import partial

import mrc
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from opentelemetry.trace.span import format_trace_id
from redis.exceptions import RedisError

import cudf

from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.redis_task_source_schema import RedisTaskSourceSchema
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.redis import RedisClient
from nv_ingest.util.tracing.logging import annotate_cm

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_source"
MODULE_NAMESPACE = "nv_ingest"
RedisTaskSourceLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def fetch_and_process_messages(redis_client: RedisClient, validated_config: RedisTaskSourceSchema):
    """Fetch messages from the Redis list and process them."""

    while True:
        try:
            job_payload = redis_client.fetch_message(validated_config.task_queue)
            ts_fetched = datetime.now()
            yield process_message(job_payload, ts_fetched)  # process_message remains unchanged
        except RedisError:
            continue  # Reconnection will be attempted on the next fetch
        except Exception as err:
            logger.error(
                f"Irrecoverable error occurred during message processing, likely malformed JOB structure: {err}"
            )
            traceback.print_exc()


def process_message(job_payload: str, ts_fetched: datetime) -> ControlMessage:
    """
    Fetch messages from the Redis list (task queue) and yield as ControlMessage.
    """
    ts_entry = datetime.now()

    job = json.loads(job_payload)
    # no_payload = copy.deepcopy(job)
    # no_payload["job_payload"]["content"] = ["[...]"]  # Redact the payload for logging
    # logger.debug("Job: %s", json.dumps(no_payload, indent=2))
    control_message = ControlMessage()
    try:
        validate_ingest_job(job)
        job_id = job.pop("job_id")
        job_payload = job.pop("job_payload", {})
        job_tasks = job.pop("tasks", [])

        tracing_options = job.pop("tracing_options", {})
        do_trace_tagging = tracing_options.get("trace", False)
        ts_send = tracing_options.get("ts_send", None)
        if ts_send is not None:
            # ts_send is in nanoseconds
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
            # logger.debug("Tasks: %s", json.dumps(task, indent=2))
            control_message.add_task(task["type"], task["task_properties"])

        # Debug Tracing
        if do_trace_tagging:
            ts_exit = datetime.now()
            control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
            control_message.set_timestamp(f"trace::entry::{MODULE_NAME}", ts_entry)
            control_message.set_timestamp(f"trace::exit::{MODULE_NAME}", ts_exit)

            if ts_send is not None:
                control_message.set_timestamp("trace::entry::redis_source_network_in", ts_send)
                control_message.set_timestamp("trace::exit::redis_source_network_in", ts_fetched)

            if trace_id is not None:
                # C++ layer in set_metadata errors out due to size of trace_id if it's an integer.
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
def _redis_task_source(builder: mrc.Builder):
    """
    A module for receiving messages from a Redis channel, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """

    validated_config = fetch_and_validate_module_config(builder, RedisTaskSourceSchema)

    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is 0, make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    _fetch_and_process_messages = partial(
        fetch_and_process_messages,
        redis_client=redis_client,
        validated_config=validated_config,
    )

    node = builder.make_source("fetch_messages_redis", _fetch_and_process_messages)
    node.launch_options.engines_per_pe = validated_config.progress_engines

    builder.register_module_output("output", node)
