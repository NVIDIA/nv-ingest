# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import logging
import uuid
import ray
import json
import copy
import threading
import time
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from opentelemetry.trace.span import format_trace_id
from pydantic import BaseModel

# Import from nv_ingest_api
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job

# Import clients
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

# Import Ray pipeline components
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline

# Import schema
from nv_ingest.framework.schemas.framework_message_broker_source_schema import MessageBrokerTaskSourceSchema

logger = logging.getLogger(__name__)


@ray.remote
class MessageBrokerTaskSource:
    """
    Ray actor for a message broker source that fetches messages from a message broker
    and converts them to IngestControlMessage objects.
    """

    def __init__(
        self,
        broker_client: Dict[str, Any],
        task_queue: str,
        progress_engines: int = 1,
        poll_interval: float = 0.1,
        batch_size: int = 10,
    ):
        """
        Initialize the message broker source.

        Parameters
        ----------
        broker_client : Dict[str, Any]
            Configuration for the message broker client
        task_queue : str
            Name of the queue to fetch messages from
        progress_engines : int, optional
            Number of engines per processing element
        poll_interval : float, optional
            Time in seconds to wait between polling attempts when no messages are found
        batch_size : int, optional
            Maximum number of messages to process in one batch
        """
        self.broker_client = broker_client
        self.task_queue = task_queue
        self.progress_engines = progress_engines
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.running = False
        self.downstream_queue = None
        self.client = self._create_client()
        self.message_count = 0
        self.start_time = None

    def _create_client(self):
        """Create the appropriate message broker client based on config."""
        client_type = self.broker_client["client_type"].lower()
        broker_params = self.broker_client.get("broker_params", {})

        if client_type == "redis":
            return RedisClient(
                host=self.broker_client["host"],
                port=self.broker_client["port"],
                db=broker_params.get("db", 0),
                max_retries=self.broker_client["max_retries"],
                max_backoff=self.broker_client["max_backoff"],
                connection_timeout=self.broker_client["connection_timeout"],
                use_ssl=broker_params.get("use_ssl", False),
            )
        elif client_type == "simple":
            # Start or retrieve the SimpleMessageBroker server
            max_queue_size = broker_params.get("max_queue_size", 10000)
            server_host = self.broker_client["host"]
            server_port = self.broker_client["port"]

            # Initialize SimpleMessageBroker server
            server_host = "0.0.0.0"  # Default binding
            server = SimpleMessageBroker(server_host, server_port, max_queue_size)

            # Start the server if not already running
            if not hasattr(server, "server_thread") or not server.server_thread.is_alive():
                server_thread = threading.Thread(target=server.serve_forever)
                server_thread.daemon = True
                server.server_thread = server_thread
                server_thread.start()
                logger.info(f"Started SimpleMessageBroker server on {server_host}:{server_port}")
            else:
                logger.info(f"SimpleMessageBroker server already running on {server_host}:{server_port}")

            return SimpleClient(
                host=server_host,
                port=server_port,
                max_retries=self.broker_client["max_retries"],
                max_backoff=self.broker_client["max_backoff"],
                connection_timeout=self.broker_client["connection_timeout"],
            )
        else:
            raise ValueError(f"Unsupported client_type: {client_type}")

    def process_message(self, job: Dict, ts_fetched: datetime) -> IngestControlMessage:
        """
        Process a job and return an IngestControlMessage.
        This reuses the logic from the original Morpheus implementation.

        Parameters
        ----------
        job : Dict
            Raw job data from the message broker
        ts_fetched : datetime
            Timestamp when the message was fetched

        Returns
        -------
        IngestControlMessage
            Processed control message
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
            control_message.set_metadata("timestamp", datetime.now().timestamp())

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
                control_message.set_timestamp("trace::entry::message_broker_task_source", ts_entry)
                control_message.set_timestamp("trace::exit::message_broker_task_source", ts_exit)

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

    def fetch_message(self, timeout=100):
        """
        Fetch a message from the message broker. This is now a local method.
        """
        try:
            logger.info(f"Attempting to fetch message from queue '{self.task_queue}'")
            job = self.client.fetch_message(self.task_queue, timeout)

            if job is None:
                logger.debug(f"No message received from '{self.task_queue}'")
                return None

            logger.info(f"Received message type: {type(job)}")

            if isinstance(job, BaseModel):
                logger.info(f"Message is a BaseModel with response_code: {job.response_code}")
                if job.response_code != 0:
                    return None
                job = json.loads(job.response)

            logger.info(f"Successfully fetched message with job_id: {job.get('job_id', 'unknown')}")
            return job
        except TimeoutError:
            logger.debug("Timeout waiting for message")
            return None
        except Exception as err:
            logger.exception(f"Error during message fetching: {err}")
            return None

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle):
        """Set the output queue for this source."""
        self.downstream_queue = queue_handle
        return True

    @ray.method(num_returns=1)
    def start(self):
        if self.running:
            return False

        self.running = True
        self.start_time = time.time()
        self.message_count = 0

        self.fetch_thread = threading.Thread(target=lambda: asyncio.run(self._fetch_messages()))
        self.fetch_thread.daemon = True  # Ensures the thread won’t block process exit
        self.fetch_thread.start()
        logger.info("Message fetch task started")

        return True

    @ray.method(num_returns=1)
    def stop(self):
        """Stop the source."""
        self.running = False
        return True

    async def _fetch_messages(self):
        import asyncio

        while self.running:
            try:
                # Fetch message using the local method.
                job = self.fetch_message()
                logger.info(f"Fetch attempt result: {'message received' if job else 'no message'}")

                if job:
                    ts_fetched = datetime.now()
                    control_message = self.process_message(job, ts_fetched)
                    if self.downstream_queue:
                        # Put the processed message into the downstream queue.
                        await self.downstream_queue.put.remote(control_message)
                    self.message_count += 1
                    logger.info(f"Processed message {self.message_count}")
                else:
                    await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.exception(f"Error fetching/processing message: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info(f"Source stopped after processing {self.message_count} messages")
        return self.message_count

    @ray.method(num_returns=1)
    def get_stats(self):
        """Get current statistics."""
        if not self.start_time:
            return {"message_count": 0}

        elapsed = time.time() - self.start_time
        return {
            "message_count": self.message_count,
            "elapsed_seconds": elapsed,
            "messages_per_second": self.message_count / elapsed if elapsed > 0 else 0,
        }


def create_message_broker_source(pipeline, config):
    """
    Create and configure a message broker source for a Ray pipeline.

    Parameters
    ----------
    pipeline : RayPipeline
        The pipeline to add the source to
    config : MessageBrokerTaskSourceSchema
        Configuration for the message broker

    Returns
    -------
    pipeline : RayPipeline
        The updated pipeline
    """
    # Add the source to the pipeline
    pipeline.add_source(
        "message_broker_source",
        MessageBrokerTaskSource,
        progress_engines=config.progress_engines,
        broker_client={
            "client_type": config.broker_client.client_type,
            "host": config.broker_client.host,
            "port": config.broker_client.port,
            "max_retries": config.broker_client.max_retries,
            "max_backoff": config.broker_client.max_backoff,
            "connection_timeout": config.broker_client.connection_timeout,
            "broker_params": config.broker_client.broker_params,
        },
        task_queue=config.task_queue,
    )

    return pipeline


def message_broker_task_source(pipeline: RayPipeline, config_dict: Dict[str, Any] = None) -> RayPipeline:
    """
    Add a message broker task source to the pipeline.

    Parameters
    ----------
    pipeline : RayPipeline
        The Ray pipeline to add the source to
    config_dict : Dict[str, Any], optional
        Configuration dictionary, will be validated against MessageBrokerTaskSourceSchema

    Returns
    -------
    RayPipeline
        The updated pipeline
    """
    # Validate configuration
    if config_dict is None:
        raise ValueError("Configuration must be provided for message_broker_task_source")

    config = MessageBrokerTaskSourceSchema.parse_obj(config_dict)

    # Add the source to the pipeline
    pipeline = create_message_broker_source(pipeline, config)

    return pipeline
