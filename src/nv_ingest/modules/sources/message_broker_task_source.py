import logging
import traceback
from datetime import datetime
from functools import partial
from typing import Dict
import copy
import json

import cudf
import mrc
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from opentelemetry.trace.span import format_trace_id

from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.message_broker_source_schema import MessageBrokerTaskSourceSchema
from nv_ingest.util.message_brokers.redis.redis_client import RedisClient
from nv_ingest.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing.logging import annotate_cm

logger = logging.getLogger(__name__)

MODULE_NAME = "message_broker_task_source"
MODULE_NAMESPACE = "nv_ingest"
MessageBrokerTaskSourceLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def fetch_and_process_messages(client, validated_config: MessageBrokerTaskSourceSchema):
    """Fetch messages from the message broker and process them."""

    while True:
        try:
            job = client.fetch_message(validated_config.task_queue, 0)
            ts_fetched = datetime.now()
            yield process_message(job, ts_fetched)  # process_message remains unchanged
        except Exception as err:
            logger.error(
                f"Irrecoverable error occurred during message processing, likely malformed JOB structure: {err}"
            )
            traceback.print_exc()
            continue  # Continue fetching the next message


def process_message(job: Dict, ts_fetched: datetime) -> ControlMessage:
    """
    Process a job and return a ControlMessage.
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
                control_message.set_timestamp("trace::entry::broker_source_network_in", ts_send)
                control_message.set_timestamp("trace::exit::broker_source_network_in", ts_fetched)

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
def _message_broker_task_source(builder: mrc.Builder):
    """
    A module for receiving messages from a message broker, converting them into DataFrames,
    and attaching job IDs to ControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.
    """

    validated_config = fetch_and_validate_module_config(builder, MessageBrokerTaskSourceSchema)

    # Determine the client type and create the appropriate client
    client_type = validated_config.client_type.lower()

    if (client_type == "redis"):
        client = RedisClient(
            host=validated_config.broker_client.host,
            port=validated_config.broker_client.port,
            db=validated_config.broker_client.broker_params.db,
            max_retries=validated_config.broker_client.max_retries,
            max_backoff=validated_config.broker_client.max_backoff,
            connection_timeout=validated_config.broker_client.connection_timeout,
            use_ssl=validated_config.broker_client.broker_params.use_ssl,
        )
    elif (client_type == "simple"):
        client = SimpleClient(
            host=validated_config.broker_client.host,
            port=validated_config.broker_client.port,
            max_retries=validated_config.broker_client.max_retries,
            max_backoff=validated_config.broker_client.max_backoff,
            connection_timeout=validated_config.broker_client.connection_timeout,
        )
    else:
        raise ValueError(f"Unsupported client_type: {client_type}")

    _fetch_and_process_messages = partial(
        fetch_and_process_messages,
        client=client,
        validated_config=validated_config,
    )

    node = builder.make_source("fetch_messages_broker", _fetch_and_process_messages)
    node.launch_options.engines_per_pe = validated_config.progress_engines

    builder.register_module_output("output", node)
