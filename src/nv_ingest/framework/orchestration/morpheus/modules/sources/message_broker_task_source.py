import logging
import uuid
from datetime import datetime
from functools import partial
from typing import Dict
import copy
import json
import threading

import mrc
import pandas as pd
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from opentelemetry.trace.span import format_trace_id
from pydantic import BaseModel

from nv_ingest.framework.orchestration.morpheus.util.modules.config_validator import (
    fetch_and_validate_module_config,
)
from nv_ingest.framework.schemas.framework_message_broker_source_schema import MessageBrokerTaskSourceSchema
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job

# Import the clients
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient

# Import the SimpleMessageBroker server
from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)

MODULE_NAME = "message_broker_task_source"
MODULE_NAMESPACE = "nv_ingest"
MessageBrokerTaskSourceLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def fetch_and_process_messages(client, validated_config: MessageBrokerTaskSourceSchema):
    """
    Fetch messages from the message broker and process them.

    Parameters
    ----------
    client : MessageBrokerClientBase
        The client used to interact with the message broker.
    validated_config : MessageBrokerTaskSourceSchema
        The validated configuration for the message broker.

    Yields
    ------
    IngestControlMessage
        The processed control message for each fetched job.

    Raises
    ------
    Exception
        If an irrecoverable error occurs during message processing.
    """

    while True:
        try:
            job = client.fetch_message(validated_config.task_queue, 10, override_fetch_mode=FetchMode.DESTRUCTIVE)
            logger.debug(f"Received Job Type: {type(job)}")
            if isinstance(job, BaseModel):
                if job.response_code != 0:
                    continue

                logger.debug("Received ResponseSchema, converting to dict")
                job = json.loads(job.response)
            else:
                logger.debug("Received something not a ResponseSchema")

            ts_fetched = datetime.now()
            yield process_message(job, ts_fetched)
        except TimeoutError:
            continue
        except Exception as err:
            logger.exception(
                f"Irrecoverable error occurred during message processing, likely malformed JSON JOB structure: {err}"
            )
            continue  # Continue fetching the next message


def process_message(job: Dict, ts_fetched: datetime) -> IngestControlMessage:
    """
    Process a job and return an IngestControlMessage.
    """
    control_message = IngestControlMessage()
    job_id = None
    try:
        if logger.isEnabledFor(logging.DEBUG):
            no_payload = copy.deepcopy(job)
            if "content" in no_payload.get("job_payload", {}):
                no_payload["job_payload"]["content"] = ["[...]"]  # Redact the payload for logging
            logger.debug("Job: %s", json.dumps(no_payload, indent=2))

        validate_ingest_job(job)

        ts_entry = datetime.now()

        job_id = job.pop("job_id")
        job_payload = job.get("job_payload", {})
        job_tasks = job.get("tasks", [])

        tracing_options = job.pop("tracing_options", {})
        do_trace_tagging = tracing_options.get("trace", False)
        ts_send = tracing_options.get("ts_send", None)
        if ts_send is not None:
            # ts_send is in nanoseconds.
            ts_send = datetime.fromtimestamp(ts_send / 1e9)
        trace_id = tracing_options.get("trace_id", None)

        response_channel = f"{job_id}"

        df = pd.DataFrame(job_payload)
        control_message.payload(df)

        annotate_cm(control_message, message="Created")
        control_message.set_metadata("response_channel", response_channel)
        control_message.set_metadata("job_id", job_id)

        # For each task, build a IngestControlMessageTask instance and add it.
        for task in job_tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            task_type = task.get("type", "unknown")
            task_props = task.get("task_properties", {})
            if not isinstance(task_props, dict):
                task_props = task_props.model_dump()

            task_obj = ControlMessageTask(
                id=task_id,
                type=task_type,
                properties=task_props,
            )
            control_message.add_task(task_obj)

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
                # Convert integer trace_id if necessary.
                if isinstance(trace_id, int):
                    trace_id = format_trace_id(trace_id)
                control_message.set_metadata("trace_id", trace_id)

            control_message.set_timestamp("latency::ts_send", datetime.now())
    except Exception as e:
        logger.exception(f"Failed to process job submission: {e}")

        if job_id is not None:
            response_channel = f"{job_id}"
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
    and attaching job IDs to IngestControlMessages.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.

    Raises
    ------
    ValueError
        If an unsupported client type is provided in the configuration.
    """

    validated_config = fetch_and_validate_module_config(builder, MessageBrokerTaskSourceSchema)

    # Determine the client type and create the appropriate client
    client_type = validated_config.broker_client.client_type.lower()
    broker_params = validated_config.broker_client.broker_params or {}

    if client_type == "redis":
        client = RedisClient(
            host=validated_config.broker_client.host,
            port=validated_config.broker_client.port,
            db=broker_params.get("db", 0),
            max_retries=validated_config.broker_client.max_retries,
            max_backoff=validated_config.broker_client.max_backoff,
            connection_timeout=validated_config.broker_client.connection_timeout,
            use_ssl=broker_params.get("use_ssl", False),
        )
    elif client_type == "simple":
        # Start or retrieve the singleton SimpleMessageBroker server
        max_queue_size = broker_params.get("max_queue_size", 10000)
        server_host = validated_config.broker_client.host
        server_port = validated_config.broker_client.port

        # TODO(Devin) add config param for server_host
        server_host = "0.0.0.0"

        # Obtain the singleton instance
        server = SimpleMessageBroker(server_host, server_port, max_queue_size)

        # Start the server if not already running
        if not hasattr(server, "server_thread") or not server.server_thread.is_alive():
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True  # Allows program to exit even if thread is running
            server.server_thread = server_thread  # Attach the thread to the server instance
            server_thread.start()
            logger.info(f"Started SimpleMessageBroker server on {server_host}:{server_port}")
        else:
            logger.info(f"SimpleMessageBroker server already running on {server_host}:{server_port}")

        # Create the SimpleClient
        client = SimpleClient(
            host=server_host,
            port=server_port,
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

    node = builder.make_source("message_broker_task_source", _fetch_and_process_messages)
    node.launch_options.engines_per_pe = validated_config.progress_engines

    builder.register_module_output("output", node)
