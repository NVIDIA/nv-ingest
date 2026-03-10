# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from typing import Optional, Literal, Dict, Any, Union

import ray
import json
import copy
import threading
import time
import random
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
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging
from nv_ingest_api.util.message_brokers.qos_scheduler import QosScheduler

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
    task_queue: str = Field(
        ..., description="The base name of the queue to fetch tasks from. Derives sub-queues for fair scheduling."
    )
    poll_interval: float = Field(default=0.0, gt=0, description="Polling interval in seconds.")


@ray.remote
class MessageBrokerTaskSourceStage(RayActorSourceStage):
    """
    Ray actor source stage for a message broker task source.

    Fetches messages from a broker, processes them, and writes to the output queue.
    """

    # Use the updated config type hint
    def __init__(self, config: MessageBrokerTaskSourceConfig, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=False, stage_name=stage_name)
        self.config: MessageBrokerTaskSourceConfig  # Add a type hint for self.config

        # Sanitize config before logging to avoid leaking secrets
        _sanitized = sanitize_for_logging(config)
        self._logger.debug(
            "Initializing MessageBrokerTaskSourceStage with config: %s", _sanitized
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

        # Backoff state for graceful retries when broker is unavailable
        self._fetch_failure_count: int = 0
        self._current_backoff_sleep: float = 0.0
        self._last_backoff_log_time: float = 0.0

        # Initialize QoS scheduler. Use a simple base-queue strategy for SimpleClient.
        strategy = "simple" if isinstance(self.client, SimpleClient) else "lottery"
        self.scheduler = QosScheduler(
            self.task_queue,
            num_prefetch_threads=6,  # one per category (no-op for simple strategy)
            total_buffer_capacity=96,  # e.g., ~16 per thread
            prefetch_poll_interval=0.002,  # faster polling for responsiveness
            prefetch_non_immediate=True,  # enable prefetch for non-immediate categories
            strategy=strategy,
        )

        self._logger.info(
            "MessageBrokerTaskSourceStage initialized. Base task queue: %s | Derived queues: %s",
            self.task_queue,
            {
                "immediate": f"{self.task_queue}_immediate",
                "micro": f"{self.task_queue}_micro",
                "small": f"{self.task_queue}_small",
                "medium": f"{self.task_queue}_medium",
                "large": f"{self.task_queue}_large",
                "default": f"{self.task_queue}",
            },
        )

    # --- Private helper methods ---
    def _create_client(self):
        # Access broker config via self.config.broker_client
        broker_config = self.config.broker_client
        self._logger.debug("Creating client of type: %s", broker_config.client_type)

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

    def _fetch_message(self, timeout=0):
        """
        Fetch a message from the message broker using fair scheduling across derived queues.
        This is a non-blocking sweep across all queues for the current scheduling cycle. If no
        message is found across any queue, return None so the caller can sleep briefly.
        """
        try:
            # Use scheduler to fetch next. In simple strategy this will block up to poll_interval on base queue.
            job = self.scheduler.fetch_next(self.client, timeout=self.config.poll_interval)
            if job is None:
                self._logger.debug(
                    "No message received from derived queues for base "
                    "'%s' (immediate, micro, small, medium, large, default)",
                    self.task_queue,
                )
                # Do not treat normal empty polls as failures
                self._fetch_failure_count = 0
                self._current_backoff_sleep = 0.0
                return None
            self._logger.debug("Received message type: %s", type(job))
            if isinstance(job, BaseModel):
                self._logger.debug("Message is a BaseModel with response_code: %s", job.response_code)
                if job.response_code not in (0, 2):
                    self._logger.debug("Message received with unhandled response_code, returning None")
                    return None
                if job.response_code == 2:
                    self._logger.debug("Message response_code == 2, returning None")
                    return None
                job = json.loads(job.response)
            self._logger.debug("Successfully fetched message with job_id: %s", job.get("job_id", "unknown"))
            # Success: reset backoff state
            self._fetch_failure_count = 0
            self._current_backoff_sleep = 0.0
            return job
        except TimeoutError:
            self._logger.debug("Timeout waiting for message")
            # Timeout is not a connectivity failure; do not escalate backoff
            return None
        except Exception as err:
            # Connectivity or other fetch issue: apply graceful backoff and avoid stacktrace spam
            self._fetch_failure_count += 1

            # Compute exponential backoff with jitter, capped by configured max_backoff
            try:
                max_backoff = getattr(self.config.broker_client, "max_backoff", 5.0)
            except Exception:
                max_backoff = 5.0
            # Start from 0.5s, double each failure
            base = 0.5
            backoff_no_jitter = min(max_backoff, base * (2 ** (self._fetch_failure_count - 1)))
            jitter = random.uniform(0, backoff_no_jitter * 0.2)
            self._current_backoff_sleep = backoff_no_jitter + jitter

            now = time.time()
            # Throttle warning logs to at most once per 5 seconds to avoid spam
            if now - self._last_backoff_log_time >= 5.0:
                self._logger.warning(
                    "Broker fetch failed (%d consecutive failures). Backing off for %.2fs. Error: %s",
                    self._fetch_failure_count,
                    self._current_backoff_sleep,
                    err,
                )
                self._last_backoff_log_time = now
            else:
                self._logger.debug(
                    "Broker fetch failed (%d). Backoff %.2fs. Error: %s",
                    self._fetch_failure_count,
                    self._current_backoff_sleep,
                    err,
                )
            return None

    def _read_input(self) -> any:
        """
        Source stage's implementation of get_input.
        Instead of reading from an input edge, fetch a message from the broker.
        """
        self._logger.debug("read_input: calling _fetch_message()")
        # Perform a non-blocking sweep across all queues for this cycle
        job = self._fetch_message(timeout=0)
        if job is None:
            # Sleep for either the configured poll interval or the current backoff, whichever is larger
            sleep_time = max(self.config.poll_interval, getattr(self, "_current_backoff_sleep", 0.0))
            self._logger.debug(
                "read_input: No job received; sleeping %.2fs (poll_interval=%.2fs, backoff=%.2fs)",
                sleep_time,
                self.config.poll_interval,
                getattr(self, "_current_backoff_sleep", 0.0),
            )
            time.sleep(sleep_time)
            # Reset one-shot backoff so that repeated failures recompute progressively
            self._current_backoff_sleep = 0.0

            return None

        self.stats["successful_queue_reads"] += 1

        ts_fetched = datetime.now()
        self._logger.debug("read_input: Job fetched, processing message")
        control_message = self._process_message(job, ts_fetched)
        self._logger.debug("read_input: Message processed, returning control message")

        return control_message

    # In the processing loop, instead of checking a boolean, we wait on the event.
    def _processing_loop(self) -> None:
        """
        Custom processing loop for a source stage.
        This loop fetches messages from the broker and writes them to the output queue,
        but blocks on the pause event when the stage is paused.
        """
        self._logger.debug("Processing loop started")
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

                # Block until not paused using the pause event.
                if self.output_queue is not None:
                    self._logger.debug("Waiting for stage to resume if paused...")

                    if not self._pause_event.is_set():
                        self._active_processing = False
                        self._pause_event.wait()  # Block if paused
                        self._active_processing = True

                    object_ref_to_put = None
                    try:
                        # Get the handle of the queue actor to set it as the owner.
                        owner_actor = self.output_queue.actor

                        # Put the object into Plasma, transferring ownership.
                        object_ref_to_put = ray.put(control_message, _owner=owner_actor)

                        # Now that the object is safely in Plasma, delete the large local copy.
                        del control_message

                        # This loop will retry indefinitely until the ObjectRef is put successfully.
                        is_put_successful = False
                        while not is_put_successful:
                            try:
                                self.output_queue.put(object_ref_to_put)
                                self.stats["successful_queue_writes"] += 1
                                is_put_successful = True  # Exit retry loop on success
                            except Exception:
                                self._logger.warning("Output queue full, retrying put()...")
                                self.stats["queue_full"] += 1
                                time.sleep(0.1)
                    finally:
                        # After the operation, delete the local ObjectRef.
                        # The primary reference is now held by the queue actor.
                        if object_ref_to_put is not None:
                            del object_ref_to_put

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

        self._logger.debug("Processing loop ending")

    @ray.method(num_returns=1)
    def start(self) -> bool:
        if self._running:
            self._logger.warning("Start called but stage is already running.")
            return False
        self._running = True
        self.start_time = time.time()
        self._message_count = 0
        self._logger.debug("Starting processing loop thread.")
        threading.Thread(target=self._processing_loop, daemon=True).start()
        self._logger.debug("MessageBrokerTaskSourceStage started.")
        return True

    @ray.method(num_returns=1)
    def stop(self) -> bool:
        self._running = False
        self._logger.debug("Stop called on MessageBrokerTaskSourceStage")
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
        self._logger.debug("Output queue set: %s", queue_handle)
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
        self._logger.debug("Stage paused.")

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
        self._logger.debug("Stage resumed.")
        return True

    @ray.method(num_returns=1)
    def swap_queues(self, new_queue: any) -> bool:
        """
        Swap in a new output queue for this stage.
        This method pauses the stage, waits for any current processing to finish,
        replaces the output queue, and then resumes the stage.
        """
        self._logger.debug("Swapping output queue: pausing stage first.")
        self.pause()
        self.set_output_queue(new_queue)
        self._logger.debug("Output queue swapped. Resuming stage.")
        self.resume()
        return True
