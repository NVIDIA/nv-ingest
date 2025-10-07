# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import json
from datetime import datetime
from typing import Optional, Dict, Any

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import validate_ingest_job
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from ..meta.python_stage_base import PythonStage
import pandas as pd
import uuid

# Centralized schemas
from nv_ingest.framework.schemas.python_message_broker_task_source_config import (
    PythonMessageBrokerTaskSourceConfig,
)

logger = logging.getLogger(__name__)


class PythonMessageBrokerTaskSource(PythonStage):
    """Python-based message broker task source.

    Fetches messages from a Simple broker, processes them, and provides them to the pipeline.
    This is a simplified version without Ray dependencies.
    """

    def __init__(self, config: PythonMessageBrokerTaskSourceConfig, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        self.config = config
        self._logger = logger
        self._logger.debug("Initializing PythonMessageBrokerTaskSource with config: %s", config)

        # Mark this as a source stage for streaming pipeline
        self.mark_as_source_stage()

        # Access validated configuration directly via self.config
        self.poll_interval = self.config.poll_interval
        self.task_queue = self.config.task_queue

        # Create the Simple client
        self.client = self._create_client()

        # Other initializations
        self._message_count = 0
        self._last_message_count = 0
        self.start_time = None
        self._running = False
        self._thread = None

        # Threading event for pause/resume functionality
        self._pause_event = threading.Event()
        self._pause_event.set()  # Initially not paused

        self._logger.debug("PythonMessageBrokerTaskSource initialized. Task queue: %s", self.task_queue)

    @nv_ingest_node_failure_try_except(annotation_id="message_broker_task_source", raise_on_failure=False)
    @traceable()
    def on_data(self, control_message: Any) -> Optional[Any]:
        """
        For source stages, this method is called by the streaming pipeline to generate messages.
        The control_message parameter is ignored for source stages.
        """
        return self.get_message()

    def _generate_source_message(self) -> Optional[Any]:
        """
        Generate a message for the streaming pipeline.
        This method is called by the streaming processing loop.
        """
        return self.get_message()

    def _create_client(self):
        """Create the Simple broker client."""
        broker_config = self.config.broker_client

        self._logger.info(f"Creating SimpleClient connection to {broker_config.host}:{broker_config.port}")

        client = SimpleClient(
            host=broker_config.host,
            port=broker_config.port,
            max_retries=broker_config.max_retries,
            max_backoff=broker_config.max_backoff,
            connection_timeout=broker_config.connection_timeout,
            interface_type=getattr(broker_config, "interface_type", "auto"),
        )

        self._logger.info(f"SimpleClient created successfully for {broker_config.host}:{broker_config.port}")
        return client

    def _process_message(self, job: dict, ts_fetched: datetime) -> IngestControlMessage:
        """
        Process a raw job fetched from the message broker into an IngestControlMessage.
        """
        control_message = IngestControlMessage()
        job_id = None

        try:
            job_id = job.get("job_id", "unknown")
            self._logger.info(f"Processing message with job_id: {job_id}")

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
                        from opentelemetry.trace.span import format_trace_id

                        trace_id = format_trace_id(trace_id)
                    control_message.set_metadata("trace_id", trace_id)

                control_message.set_timestamp("latency::ts_send", datetime.now())

            self._logger.info(f"Successfully processed message {job_id} into IngestControlMessage")
            return control_message

        except Exception as e:
            job_id = job.get("job_id", "unknown") if job else "unknown"
            self._logger.error(f"Failed to process message {job_id}: {e}")

            if job_id is not None:
                response_channel = f"{job_id}"
                control_message.set_metadata("job_id", job_id)
                control_message.set_metadata("response_channel", response_channel)
                control_message.set_metadata("cm_failed", True)

                annotate_cm(control_message, message="Failed to process job submission", error=str(e))
            else:
                raise

            return control_message

    def _fetch_message(self, timeout=100) -> Optional[Dict[str, Any]]:
        """
        Fetch a message from the message broker.
        """
        try:
            self._logger.debug(f"Fetching message from queue: {self.task_queue}")
            response = self.client.fetch_message(self.task_queue, timeout=(timeout, None))

            if response.response_code == 200 and response.response:
                # Parse response data if it's a string
                response_data = response.response
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except json.JSONDecodeError as e:
                        self._logger.error(f"Failed to parse JSON response: {e}")
                        return None

                job_id = response_data.get("job_id", "unknown") if isinstance(response_data, dict) else "unknown"
                self._logger.info(f"Successfully fetched message {job_id} from queue {self.task_queue}")
                return response_data
            elif response.response_code == 0 and response.response:  # Success with message
                # Parse response data if it's a string
                response_data = response.response
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except json.JSONDecodeError as e:
                        self._logger.error(f"Failed to parse JSON response: {e}")
                        return None

                job_id = response_data.get("job_id", "unknown") if isinstance(response_data, dict) else "unknown"
                self._logger.info(f"Successfully fetched message {job_id} from queue {self.task_queue} (code 0)")
                return response_data
            elif response.response_code == 204:  # No content
                self._logger.debug(f"No messages available in queue {self.task_queue} (204)")
                return None
            elif response.response_code == 2:  # Job not ready (empty queue)
                self._logger.debug(f"No messages available in queue {self.task_queue} (2 - Job not ready)")
                return None
            else:
                self._logger.warning(f"Unexpected response code: {response.response_code} from queue {self.task_queue}")
                self._logger.warning(f"Response reason: {getattr(response, 'response_reason', 'N/A')}")
                self._logger.warning(f"Response data: {getattr(response, 'response', 'N/A')}")
                return None

        except Exception as e:
            self._logger.error(f"Failed to fetch message from queue {self.task_queue}: {e}")
            return None

    def get_message(self) -> Optional[IngestControlMessage]:
        """
        Get the next message from the broker.
        This is the main interface for the pipeline to fetch messages.
        """
        if not self._pause_event.is_set():
            self._logger.debug("Source is paused, not fetching messages")
            return None

        job = self._fetch_message()
        if job is None:
            self._logger.debug("No message available from broker")
            return None

        ts_fetched = datetime.now()
        control_message = self._process_message(job, ts_fetched)
        self._message_count += 1

        self._logger.info(f"Successfully obtained and processed message (total processed: {self._message_count})")
        return control_message

    def start(self):
        """Start the source."""
        self._running = True
        self.start_time = datetime.now()
        self._logger.info("PythonMessageBrokerTaskSource started")
        self._logger.info(
            f"Broker connection established to {self.config.broker_client.host}:{self.config.broker_client.port}"
        )

    def stop(self):
        """Stop the source."""
        self._running = False
        self._logger.info("PythonMessageBrokerTaskSource stopped")

    def pause(self) -> bool:
        """
        Pause the source. This clears the pause event, causing message fetching to be blocked.

        Returns
        -------
        bool
            True after the source is paused.
        """
        self._pause_event.clear()
        self._logger.info("PythonMessageBrokerTaskSource paused")
        return True

    def resume(self) -> bool:
        """
        Resume the source. This sets the pause event, allowing message fetching to proceed.

        Returns
        -------
        bool
            True after the source is resumed.
        """
        self._pause_event.set()
        self._logger.info("PythonMessageBrokerTaskSource resumed")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the source."""
        return {
            "message_count": self._message_count,
            "running": self._running,
            "paused": not self._pause_event.is_set(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }
