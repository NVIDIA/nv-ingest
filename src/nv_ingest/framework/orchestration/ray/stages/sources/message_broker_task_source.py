# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import multiprocessing
import uuid
import socket

import ray
import json
import copy
import threading
import time
from datetime import datetime

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
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logging.basicConfig(level=logging.WARNING)
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
    processes them into control messages, and writes them to the output queue.

    As a source stage, it overrides read_input() to use its own message-fetching logic.
    """

    def __init__(self, config: BaseModel, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        logger.debug("Initializing MessageBrokerTaskSourceStage with config: %s", config)
        # Configuration specific to message broker task source.
        self.broker_client = self.config.broker_client
        self.task_queue = self.config.task_queue
        self.poll_interval = self.config.poll_interval
        self.client = self._create_client()
        self.message_count = 0
        self.start_time = None
        # For source stages, output is provided via a direct queue.
        self.output_queue = None
        logger.debug("MessageBrokerTaskSourceStage initialized. Task queue: %s", self.task_queue)

    # --- Private helper methods ---
    def _create_client(self):
        client_type = self.broker_client["client_type"].lower()
        broker_params = self.broker_client.get("broker_params", {})
        logger.debug("Creating client of type: %s", client_type)
        if client_type == "redis":
            client = RedisClient(
                host=self.broker_client["host"],
                port=self.broker_client["port"],
                db=broker_params.get("db", 0),
                max_retries=self.broker_client["max_retries"],
                max_backoff=self.broker_client["max_backoff"],
                connection_timeout=self.broker_client["connection_timeout"],
                use_ssl=broker_params.get("use_ssl", False),
            )
            logger.debug("RedisClient created: %s", client)
            return client
        elif client_type == "simple":
            client = SimpleClient(
                host=self.broker_client["host"],
                port=self.broker_client["port"],
                max_retries=self.broker_client["max_retries"],
                max_backoff=self.broker_client["max_backoff"],
                connection_timeout=self.broker_client["connection_timeout"],
            )
            logger.debug("SimpleClient created: %s", client)
            return client
        else:
            logger.error("Unsupported client_type: %s", client_type)
            raise ValueError(f"Unsupported client_type: {client_type}")

    @staticmethod
    def _process_message(job: dict, ts_fetched: datetime) -> any:
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
                logger.debug("Processed job payload for logging: %s", json.dumps(no_payload, indent=2))

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
            logger.debug("Message processed successfully with job_id: %s", job_id)
        except Exception as e:
            logger.exception("Failed to process job submission: %s", e)
            if job_id is not None:
                response_channel = f"{job_id}"
                control_message.set_metadata("job_id", job_id)
                control_message.set_metadata("response_channel", response_channel)
                control_message.set_metadata("cm_failed", True)
                annotate_cm(control_message, message="Failed to process job submission", error=str(e))
            else:
                raise

        return control_message

    def _fetch_message(self, timeout=100):
        """
        Fetch a message from the message broker.
        """
        logger.info("Attempting to fetch message from queue '%s'", self.task_queue)
        try:
            job = self.client.fetch_message(self.task_queue, timeout)
            if job is None:
                logger.debug("No message received from '%s'", self.task_queue)
                return None
            logger.info("Received message type: %s", type(job))
            if isinstance(job, BaseModel):
                logger.info("Message is a BaseModel with response_code: %s", job.response_code)
                if job.response_code != 0:
                    logger.debug("Message response_code != 0, returning None")
                    return None
                job = json.loads(job.response)
            logger.info("Successfully fetched message with job_id: %s", job.get("job_id", "unknown"))
            return job
        except TimeoutError:
            logger.debug("Timeout waiting for message")
            return None
        except Exception as err:
            logger.exception("Error during message fetching: %s", err)
            return None

    def read_input(self) -> any:
        """
        Source stage's implementation of get_input.
        Instead of reading from an input edge, fetch a message from the broker.
        """
        logger.debug("read_input: calling _fetch_message()")
        job = self._fetch_message(timeout=100)
        if job is None:
            logger.debug("read_input: No job received, sleeping for poll_interval: %s", self.config.poll_interval)
            time.sleep(self.config.poll_interval)
            return None
        ts_fetched = datetime.now()
        logger.debug("read_input: Job fetched, processing message")
        control_message = self._process_message(job, ts_fetched)
        logger.debug("read_input: Message processed, returning control message")
        return control_message

    def on_data(self, control_message: any) -> any:
        """
        Process the control message.
        For this source stage, no additional processing is done, so simply return it.
        """
        logger.debug("on_data: Received control message for processing")
        return control_message

    def _processing_loop(self) -> None:
        logger.info("Processing loop started")
        iteration = 0
        while self.running:
            iteration += 1
            try:
                logger.debug("Processing loop iteration: %s", iteration)
                control_message = self.read_input()
                if control_message is None:
                    logger.debug(
                        "No control message received; sleeping for poll_interval: %s", self.config.poll_interval
                    )
                    time.sleep(self.config.poll_interval)
                    continue
                logger.debug("Control message received; processing data")
                updated_cm = self.on_data(control_message)
                if (updated_cm is not None) and (self.output_queue is not None):
                    self.output_queue.put(updated_cm)
                self.stats["processed"] += 1
                self.message_count += 1
                logger.debug(
                    "Processing loop iteration %s complete. Total processed: %s", iteration, self.stats["processed"]
                )
            except Exception as e:
                logger.exception("Error in processing loop at iteration %s: %s", iteration, e)
                time.sleep(self.config.poll_interval)

    @ray.method(num_returns=1)
    def start(self) -> bool:
        if self.running:
            logger.info("Start called but stage is already running.")
            return False
        self.running = True
        self.start_time = time.time()
        self.message_count = 0
        logger.info("Starting processing loop thread.")
        threading.Thread(target=self._processing_loop, daemon=True).start()
        logger.info("Message broker task source stage started")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        self.running = False
        logger.info("Stop called on MessageBrokerTaskSourceStage")
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        stats = {
            "message_count": self.message_count,
            "elapsed_seconds": elapsed,
            "messages_per_second": self.message_count / elapsed if elapsed > 0 else 0,
        }
        logger.info("get_stats: %s", stats)
        return stats

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        self.output_queue = queue_handle
        logger.info("Output queue set: %s", queue_handle)
        return True


def start_simple_message_broker(broker_client: dict) -> multiprocessing.Process:
    """
    Starts a SimpleMessageBroker server in a separate process.

    Parameters
    ----------
    broker_client : dict
        Broker configuration. Expected keys include:
          - "port": the port to bind the server to,
          - "broker_params": optionally including "max_queue_size",
          - and any other parameters required by SimpleMessageBroker.

    Returns
    -------
    multiprocessing.Process
        The process running the SimpleMessageBroker server.
    """

    def broker_server():
        from nv_ingest_api.util.message_brokers.simple_message_broker.broker import SimpleMessageBroker

        # Use max_queue_size from broker_params or default to 10000.
        broker_params = broker_client.get("broker_params", {})
        max_queue_size = broker_params.get("max_queue_size", 10000)
        server_host = broker_client.get("host", "0.0.0.0")
        server_port = broker_client.get("port", 7671)
        # Optionally, set socket options here for reuse.
        server = SimpleMessageBroker(server_host, server_port, max_queue_size)
        # Enable address reuse on the server socket.
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.serve_forever()

    p = multiprocessing.Process(target=broker_server)
    p.daemon = True
    p.start()
    logger.info(f"Started SimpleMessageBroker server in separate process on port {broker_client['port']}")
    return p
