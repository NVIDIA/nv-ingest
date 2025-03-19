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
from typing import Any

import pandas as pd
from opentelemetry.trace.span import format_trace_id
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage

# Import from nv_ingest_api
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job

# Import clients
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)


class MessageBrokerTaskSourceConfig(BaseModel):
    broker_client: dict
    task_queue: str
    poll_interval: float = 0.1


@ray.remote
class MessageBrokerTaskSourceStage(RayActorSourceStage):
    """
    Ray actor source stage for a message broker task source.

    This stage fetches messages from a message broker (via Redis or a simple broker),
    processes them into control messages, and writes them to the output edge.

    As a source stage, it overrides get_input() to use its own message-fetching logic.
    """

    def __init__(self, config: BaseModel, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        # Configuration specific to message broker task source.
        self.broker_client = self.config.broker_client
        self.task_queue = self.config.task_queue
        self.poll_interval = self.config.poll_interval
        self.client = self._create_client()
        self.message_count = 0
        self.start_time = None

    # --- Private helper methods ---
    def _create_client(self):
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
            max_queue_size = broker_params.get("max_queue_size", 10000)
            server_host = self.broker_client["host"]
            server_port = self.broker_client["port"]

            # Bind to all interfaces.
            server_host = "0.0.0.0"
            server = SimpleMessageBroker(server_host, server_port, max_queue_size)

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

    @staticmethod
    def _process_message(job: dict, ts_fetched: datetime) -> Any:
        """
        Process a raw job fetched from the message broker into an IngestControlMessage.
        """
        control_message = IngestControlMessage()
        job_id = None
        try:
            if logger.isEnabledFor(logging.DEBUG):
                no_payload = copy.deepcopy(job)
                if "content" in no_payload.get("job_payload", {}):
                    no_payload["job_payload"]["content"] = ["[...]"]
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
                ts_send = datetime.fromtimestamp(ts_send / 1e9)
            trace_id = tracing_options.get("trace_id", None)
            response_channel = f"{job_id}"

            df = pd.DataFrame(job_payload)
            control_message.payload(df)
            annotate_cm(control_message, message="Created")
            control_message.set_metadata("response_channel", response_channel)
            control_message.set_metadata("job_id", job_id)
            control_message.set_metadata("timestamp", datetime.now().timestamp())

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

            if do_trace_tagging:
                ts_exit = datetime.now()
                control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
                control_message.set_timestamp("trace::entry::message_broker_task_source", ts_entry)
                control_message.set_timestamp("trace::exit::message_broker_task_source", ts_exit)
                if ts_send is not None:
                    control_message.set_timestamp("trace::entry::broker_source_network_in", ts_send)
                    control_message.set_timestamp("trace::exit::broker_source_network_in", ts_fetched)
                if trace_id is not None:
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

        logger.warning(f"[SOURCE CM]: {control_message.payload()}\n")
        return control_message

    def _fetch_message(self, timeout=100):
        """
        Fetch a message from the message broker.
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

    async def read_input(self) -> Any:
        """
        Source stage's implementation of get_input.
        Instead of reading from an input edge, fetch a message from the broker.
        """
        job = self._fetch_message(timeout=100)
        if job is None:
            await asyncio.sleep(self.config.poll_interval)
            return None
        ts_fetched = datetime.now()
        control_message = self._process_message(job, ts_fetched)
        return control_message

    async def on_data(self, control_message: Any) -> Any:
        """
        Process the control message.
        For this source stage, no additional processing is done, so simply return it.
        """
        return control_message

    @ray.method(num_returns=1)
    def start(self) -> bool:
        if self.running:
            return False
        self.running = True
        self.start_time = time.time()
        self.message_count = 0
        threading.Thread(target=lambda: asyncio.run(self._processing_loop()), daemon=True).start()
        logger.info("Message broker task source stage started")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        self.running = False
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "message_count": self.message_count,
            "elapsed_seconds": elapsed,
            "messages_per_second": self.message_count / elapsed if elapsed > 0 else 0,
        }

    @ray.method(num_returns=1)
    def set_output_edge(self, edge_handle: Any) -> bool:
        self.output_edge = edge_handle
        return True
