# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
import uuid
import socket
from typing import Optional, Literal, Dict, Any, Union

import ray
import json
import copy
import threading
import time
from datetime import datetime

import pandas as pd
from opentelemetry.trace.span import format_trace_id
from pydantic import BaseModel, Field

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage

# Import from nv_ingest_api
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job

# Import clients
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)


class BrokerParamsRedis(BaseModel):
    """Specific parameters for Redis broker_params."""

    db: int = 0
    use_ssl: bool = False


class BaseBrokerClientConfig(BaseModel):
    """Base configuration common to all broker clients."""

    host: str = Field(..., description="Hostname or IP address of the message broker.")
    port: int = Field(..., description="Port number of the message broker.")
    max_retries: int = Field(default=5, ge=0, description="Maximum number of connection retries.")
    max_backoff: float = Field(default=5.0, gt=0, description="Maximum backoff delay in seconds between retries.")
    connection_timeout: float = Field(default=30.0, gt=0, description="Connection timeout in seconds.")


class RedisClientConfig(BaseBrokerClientConfig):
    """Configuration specific to the Redis client."""

    client_type: Literal["redis"] = Field(..., description="Specifies the client type as Redis.")
    broker_params: BrokerParamsRedis = Field(
        default_factory=BrokerParamsRedis, description="Redis-specific parameters like db and ssl."
    )


class SimpleClientConfig(BaseBrokerClientConfig):
    """Configuration specific to the Simple client."""

    client_type: Literal["simple"] = Field(..., description="Specifies the client type as Simple.")
    broker_params: Optional[Dict[str, Any]] = Field(
        default={}, description="Optional parameters for Simple client (currently unused)."
    )


# --- Define Updated Source Configuration ---


class MessageBrokerTaskSourceConfig(BaseModel):
    """
    Configuration for the MessageBrokerTaskSourceStage.

    Attributes
    ----------
    broker_client : Union[RedisClientConfig, SimpleClientConfig]
        Configuration parameters for connecting to the message broker.
        The specific schema is determined by the 'client_type' field.
    task_queue : str
        The name of the queue to fetch tasks from.
    poll_interval : float, optional
        The polling interval (in seconds) for fetching messages. Defaults to 0.1.
    """

    # Use the discriminated union for broker_client
    broker_client: Union[RedisClientConfig, SimpleClientConfig] = Field(..., discriminator="client_type")
    task_queue: str = Field(..., description="The name of the queue to fetch tasks from.")
    poll_interval: float = Field(default=0.1, gt=0, description="Polling interval in seconds.")


@ray.remote
class MessageBrokerTaskSourceStage(RayActorSourceStage):
    """
    Ray actor source stage for a message broker task source.

    Fetches messages from a broker, processes them, and writes to the output queue.
    """

    # Use the updated config type hint
    def __init__(self, config: MessageBrokerTaskSourceConfig) -> None:
        super().__init__(config, log_to_stdout=False)
        self.config: MessageBrokerTaskSourceConfig  # Add type hint for self.config
        self._logger.debug(
            "Initializing MessageBrokerTaskSourceStage with config: %s", config.dict()
        )  # Log validated config

        # Access validated configuration directly via self.config
        self.poll_interval = self.config.poll_interval
        self.task_queue = self.config.task_queue

        # Create the client using validated config
        self.client = self._create_client()

        # Other initializations
        self._message_count = 0
        self._last_message_count = 0
        self.output_queue = None  # Presumably set later or via base class
        self.start_time = None

        # Threading event remains the same
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused

        self._logger.debug("MessageBrokerTaskSourceStage initialized. Task queue: %s", self.task_queue)

    # --- Private helper methods ---
    def _create_client(self):
        # Access broker config via self.config.broker_client
        broker_config = self.config.broker_client
        self._logger.info("Creating client of type: %s", broker_config.client_type)

        if broker_config.client_type == "redis":
            client = RedisClient(
                host=broker_config.host,
                port=broker_config.port,
                db=broker_config.broker_params.db,  # Use nested model attribute access
                max_retries=broker_config.max_retries,
                max_backoff=broker_config.max_backoff,
                connection_timeout=broker_config.connection_timeout,
                use_ssl=broker_config.broker_params.use_ssl,  # Use nested model attribute access
            )
            self._logger.debug("RedisClient created: %s", client)  # Consider logging non-sensitive parts if needed
            return client
        elif broker_config.client_type == "simple":
            server_host = broker_config.host
            server_host = "0.0.0.0"
            client = SimpleClient(
                host=server_host,  # Using configured host
                port=broker_config.port,
                max_retries=broker_config.max_retries,
                max_backoff=broker_config.max_backoff,
                connection_timeout=broker_config.connection_timeout,
            )
            self._logger.debug("SimpleClient created: %s", client)
            return client

    def _process_message(self, job: dict, ts_fetched: datetime) -> Any:
        """
        Process a raw job fetched from the message broker into an IngestControlMessage.
        """
        control_message = IngestControlMessage()
        job_id = None

        try:
            # Log the payload (with content redacted) if in debug mode
            if self._logger.isEnabledFor(logging.DEBUG):
                no_payload = copy.deepcopy(job)
                if "content" in no_payload.get("job_payload", {}):
                    no_payload["job_payload"]["content"] = ["[...]"]
                self._logger.debug("Processed job payload for logging: %s", json.dumps(no_payload, indent=2))

            # Validate incoming job structure
            validate_ingest_job(job)

            ts_entry = datetime.now()
            job_id = job.pop("job_id")

            job_payload = job.get("job_payload", {})
            job_tasks = job.get("tasks", [])
            tracing_options = job.pop("tracing_options", {})

            # Extract tracing options
            do_trace_tagging = tracing_options.get("trace", True)
            if do_trace_tagging in (True, "True", "true", "1"):
                do_trace_tagging = True

            ts_send = tracing_options.get("ts_send")
            if ts_send is not None:
                ts_send = datetime.fromtimestamp(ts_send / 1e9)
            trace_id = tracing_options.get("trace_id")

            # Create response channel and load payload
            response_channel = f"{job_id}"
            df = pd.DataFrame(job_payload)
            control_message.payload(df)
            annotate_cm(control_message, message="Created")

            # Add basic metadata
            control_message.set_metadata("response_channel", response_channel)
            control_message.set_metadata("job_id", job_id)
            control_message.set_metadata("timestamp", datetime.now().timestamp())

            # Add task definitions to the control message
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

            # Apply tracing metadata and timestamps if enabled
            control_message.set_metadata("config::add_trace_tagging", do_trace_tagging)
            if do_trace_tagging:
                ts_exit = datetime.now()

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

            self._logger.debug("Message processed successfully with job_id: %s", job_id)

        except Exception as e:
            self._logger.exception("Failed to process job submission: %s", e)

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
        try:
            job = self.client.fetch_message(self.task_queue, timeout)
            if job is None:
                self._logger.debug("No message received from '%s'", self.task_queue)
                return None
            self._logger.debug("Received message type: %s", type(job))
            if isinstance(job, BaseModel):
                self._logger.debug("Message is a BaseModel with response_code: %s", job.response_code)
                if job.response_code != 0:
                    self._logger.debug("Message response_code != 0, returning None")
                    return None
                job = json.loads(job.response)
            self._logger.debug("Successfully fetched message with job_id: %s", job.get("job_id", "unknown"))
            return job
        except TimeoutError:
            self._logger.debug("Timeout waiting for message")
            return None
        except Exception as err:
            self._logger.exception("Error during message fetching: %s", err)
            return None

    def _read_input(self) -> any:
        """
        Source stage's implementation of get_input.
        Instead of reading from an input edge, fetch a message from the broker.
        """
        self._logger.debug("read_input: calling _fetch_message()")
        job = self._fetch_message(timeout=100)
        if job is None:
            self._logger.debug("read_input: No job received, sleeping for poll_interval: %s", self.config.poll_interval)
            time.sleep(self.config.poll_interval)

            return None

        self.stats["successful_queue_reads"] += 1

        ts_fetched = datetime.now()
        self._logger.debug("read_input: Job fetched, processing message")
        control_message = self._process_message(job, ts_fetched)
        self._logger.debug("read_input: Message processed, returning control message")

        return control_message

    def on_data(self, control_message: any) -> any:
        """
        Process the control message.
        For this source stage, no additional processing is done, so simply return it.
        """
        self._logger.debug("on_data: Received control message for processing")
        return control_message

    # In the processing loop, instead of checking a boolean, we wait on the event.
    def _processing_loop(self) -> None:
        """
        Custom processing loop for a source stage.
        This loop fetches messages from the broker and writes them to the output queue,
        but blocks on the pause event when the stage is paused.
        """
        self._logger.info("Processing loop started")
        iteration = 0
        while self._running:
            iteration += 1
            try:
                self._logger.debug("Processing loop iteration: %s", iteration)
                control_message = self._read_input()
                if control_message is None:
                    self._logger.debug(
                        "No control message received; sleeping for poll_interval: %s", self.config.poll_interval
                    )
                    time.sleep(self.config.poll_interval)
                    continue

                self._active_processing = True

                self._logger.debug("Control message received; processing data")
                updated_cm = self.on_data(control_message)

                # Block until not paused using the pause event.
                if self.output_queue is not None:
                    self._logger.debug("Waiting for stage to resume if paused...")

                    if not self._pause_event.is_set():
                        self._active_processing = False
                        self._pause_event.wait()  # Block if paused
                        self._active_processing = True

                    while True:
                        try:
                            self.output_queue.put(updated_cm)
                            self.stats["successful_queue_writes"] += 1
                            break
                        except Exception:
                            self._logger.warning("Output queue full, retrying put()...")
                            self.stats["queue_full"] += 1
                            time.sleep(0.1)

                self.stats["processed"] += 1
                self._message_count += 1

                self._logger.debug(f"Sourced message_count: {self._message_count}")
                self._logger.debug("Iteration %s complete. Total processed: %s", iteration, self.stats["processed"])
            except Exception as e:
                self._logger.exception("Error in processing loop at iteration %s: %s", iteration, e)
                time.sleep(self.config.poll_interval)
            finally:
                self._active_processing = False
                self._shutdown_signal_complete = True

        self._logger.info("Processing loop ending")

    @ray.method(num_returns=1)
    def start(self) -> bool:
        if self._running:
            self._logger.info("Start called but stage is already running.")
            return False
        self._running = True
        self.start_time = time.time()
        self._message_count = 0
        self._logger.info("Starting processing loop thread.")
        threading.Thread(target=self._processing_loop, daemon=True).start()
        self._logger.info("MessageBrokerTaskSourceStage started.")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        self._running = False
        self._logger.info("Stop called on MessageBrokerTaskSourceStage")
        return True

    @ray.method(num_returns=1)
    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        delta = self._message_count - self._last_message_count
        self._last_message_count = self._message_count
        stats = {
            "active_processing": 1 if self._active_processing else 0,
            "delta_processed": delta,
            "elapsed": elapsed,
            "errors": self.stats.get("errors", 0),
            "failed": 0,
            "processed": self._message_count,
            "processing_rate_cps": self._message_count / elapsed if elapsed > 0 else 0,
            "successful_queue_reads": self.stats.get("successful_queue_reads", 0),
            "successful_queue_writes": self.stats.get("successful_queue_writes", 0),
            "queue_full": self.stats.get("queue_full", 0),
        }

        return stats

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        self.output_queue = queue_handle
        self._logger.info("Output queue set: %s", queue_handle)
        return True

    @ray.method(num_returns=1)
    def pause(self) -> bool:
        """
        Pause the stage. This clears the pause event, causing the processing loop
        to block before writing to the output queue.

        Returns
        -------
        bool
            True after the stage is paused.
        """
        self._pause_event.clear()
        self._logger.info("Stage paused.")

        return True

    @ray.method(num_returns=1)
    def resume(self) -> bool:
        """
        Resume the stage. This sets the pause event, allowing the processing loop
        to proceed with writing to the output queue.

        Returns
        -------
        bool
            True after the stage is resumed.
        """
        self._pause_event.set()
        self._logger.info("Stage resumed.")
        return True

    @ray.method(num_returns=1)
    def swap_queues(self, new_queue: any) -> bool:
        """
        Swap in a new output queue for this stage.
        This method pauses the stage, waits for any current processing to finish,
        replaces the output queue, and then resumes the stage.
        """
        self._logger.info("Swapping output queue: pausing stage first.")
        self.pause()
        self.set_output_queue(new_queue)
        self._logger.info("Output queue swapped. Resuming stage.")
        self.resume()
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
