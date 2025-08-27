# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient
from ..meta.python_stage_base import PythonStage
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest.framework.schemas.broker_client_configs import SimpleClientConfig

logger = logging.getLogger(__name__)


def _default_simple_client() -> SimpleClientConfig:
    """Provide sane defaults for libmode when not explicitly configured."""
    return SimpleClientConfig(host="0.0.0.0", port=7671)


class PythonMessageBrokerTaskSinkConfig(BaseModel):
    """Configuration for the PythonMessageBrokerTaskSink.

    Attributes
    ----------
    broker_client : SimpleClientConfig
        Configuration parameters for connecting to the Simple message broker.
    poll_interval : float, optional
        The polling interval (in seconds) for processing messages. Defaults to 0.1.
    """

    broker_client: SimpleClientConfig = Field(default_factory=_default_simple_client)
    poll_interval: float = Field(default=0.1, gt=0)


class PythonMessageBrokerTaskSink(PythonStage):
    """Python-based message broker task sink.

    Processes messages and sends results back to the Simple message broker.
    This is a simplified version without Ray dependencies.
    """

    def __init__(self, config: PythonMessageBrokerTaskSinkConfig, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        self.config = config
        self._logger = logger
        self._logger.debug("Initializing PythonMessageBrokerTaskSink with config: %s", config.dict())

        self.poll_interval = self.config.poll_interval

        # Create the Simple broker client
        self.client = self._create_client()
        self.start_time = None
        self.message_count = 0
        self._running = False

        self._logger.debug("PythonMessageBrokerTaskSink initialized")

    def _create_client(self):
        """Create the Simple broker client."""
        broker_config = self.config.broker_client

        return SimpleClient(
            host=broker_config.host,
            port=broker_config.port,
            max_retries=broker_config.max_retries,
            max_backoff=broker_config.max_backoff,
            connection_timeout=broker_config.connection_timeout,
            interface_type=broker_config.interface_type,
        )

    @staticmethod
    def _extract_data_frame(message: Any):
        """
        Extracts a DataFrame from a message payload and returns it along with selected columns.
        """
        try:
            df = message.payload()
            logger.debug(f"Sink received DataFrame with {len(df)} rows.")
            keep_cols = ["document_type", "metadata"]
            return df, df[keep_cols].to_dict(orient="records")
        except Exception as err:
            logger.error(f"Failed to extract DataFrame: {err}")
            return None, None

    @staticmethod
    def _split_large_dict(json_data: List[Dict[str, Any]], size_limit: int) -> List[List[Dict[str, Any]]]:
        """Split large JSON data into smaller chunks based on size limit."""
        chunks = []
        current_chunk = []
        current_size = 0

        for item in json_data:
            item_size = len(json.dumps(item))

            if current_size + item_size > size_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_size = item_size
            else:
                current_chunk.append(item)
                current_size += item_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_json_payload(self, message: Any, df_json: Any) -> List[Dict[str, Any]]:
        """
        Creates JSON payloads based on the message data. Splits the data if it exceeds a size limit.
        """
        df_json_str = json.dumps(df_json)
        df_json_size = sys.getsizeof(df_json_str)
        size_limit = 128 * 1024 * 1024  # 128 MB limit
        if df_json_size > size_limit:
            data_fragments = self._split_large_dict(df_json, size_limit)
            fragment_count = len(data_fragments)
        else:
            data_fragments = [df_json]
            fragment_count = 1

        ret_val_json_list = []
        for i, fragment_data in enumerate(data_fragments):
            ret_val_json = {
                "status": "success" if not message.get_metadata("cm_failed", False) else "failed",
                "description": (
                    "Successfully processed the message."
                    if not message.get_metadata("cm_failed", False)
                    else "Failed to process the message."
                ),
                "data": fragment_data,
                "fragment": i,
                "fragment_count": fragment_count,
            }
            if i == 0 and message.get_metadata("config::add_trace_tagging", True):
                trace_snapshot = message.filter_timestamp("trace::")
                ret_val_json["trace"] = {key: ts.timestamp() * 1e9 for key, ts in trace_snapshot.items()}
                ret_val_json["annotations"] = {
                    key: message.get_metadata(key) for key in message.list_metadata() if key.startswith("annotation::")
                }
            ret_val_json_list.append(ret_val_json)
        logger.debug(f"Sink created {len(ret_val_json_list)} JSON payloads.")
        return ret_val_json_list

    def _push_to_broker(self, json_payloads: List[str], response_channel: str, retry_count: int = 2) -> None:
        """
        Pushes JSON payloads to the broker channel, retrying on failure.
        """
        # Check payload sizes first (like original)
        for payload in json_payloads:
            payload_size = sys.getsizeof(payload)
            size_limit = 2**28  # 256 MB
            if payload_size > size_limit:
                raise ValueError(f"Payload size {payload_size} exceeds limit of {size_limit / 1e6} MB.")

        # Push all payloads with retry logic (like original)
        for attempt in range(retry_count):
            try:
                for payload in json_payloads:
                    self.client.submit_message(response_channel, payload)
                self._logger.debug(f"Sink forwarded message to channel '{response_channel}'.")
                return
            except ValueError as e:
                self._logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    raise

    def _handle_failure(
        self, response_channel: str, json_result_fragments: List[Dict[str, Any]], e: Exception, mdf_size: int
    ):
        """
        Handles failure by logging and pushing a failure message to the broker.
        """
        self._logger.error(f"Processing failed: {e}")

        failure_payload = {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.now().isoformat(),
            "fragments_count": len(json_result_fragments),
            "data_size": mdf_size,
        }

        try:
            self._push_to_broker([json.dumps(failure_payload)], response_channel)
        except Exception as push_error:
            self._logger.error(f"Failed to push failure message: {push_error}")

    @nv_ingest_node_failure_try_except(annotation_id="message_broker_task_sink", raise_on_failure=False)
    @traceable()
    def on_data(self, control_message: Any) -> Any:
        """
        Processes the control message and pushes the resulting JSON payloads to the broker.
        """
        mdf, df_json = None, None
        json_result_fragments = []
        response_channel = control_message.get_metadata("response_channel")
        try:
            cm_failed = control_message.get_metadata("cm_failed", False)
            if not cm_failed:
                mdf, df_json = self._extract_data_frame(control_message)
                json_result_fragments = self._create_json_payload(control_message, df_json)
            else:
                json_result_fragments = self._create_json_payload(control_message, None)

            total_payload_size = 0
            json_payloads = []
            for i, fragment in enumerate(json_result_fragments, start=1):
                payload = json.dumps(fragment)
                size_bytes = len(payload.encode("utf-8"))
                total_payload_size += size_bytes
                size_mb = size_bytes / (1024 * 1024)
                logger.debug(f"Sink Fragment {i} size: {size_mb:.2f} MB")
                json_payloads.append(payload)

            total_size_mb = total_payload_size / (1024 * 1024)
            logger.debug(f"Sink Total JSON payload size: {total_size_mb:.2f} MB")
            annotate_cm(control_message, message="Pushed")
            self._push_to_broker(json_payloads, response_channel)

        except ValueError as e:
            mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
            self._handle_failure(response_channel, json_result_fragments, e, mdf_size)
        except Exception as e:
            logger.exception(f"Critical error processing message: {e}")
            mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
            self._handle_failure(response_channel, json_result_fragments, e, mdf_size)

        self.message_count += 1
        self._logger.debug(f"[Message Broker Sink] Processed message count: {self.message_count}")

        return control_message

    def process_message(self, control_message: Any) -> bool:
        """
        Processes the control message and pushes the resulting JSON payloads to the broker.
        This is the main interface for the pipeline to send messages.
        """
        try:
            # Extract response channel from the message
            response_channel = "response_queue"  # Default channel
            if hasattr(control_message, "tasks") and control_message.tasks:
                task_props = getattr(control_message.tasks[0], "task_properties", {})
                response_channel = task_props.get("response_channel", response_channel)
                logger.info(f"Using response channel: {response_channel}")

            # Extract DataFrame and selected columns
            df, selected_columns = self._extract_data_frame(control_message)

            # Create JSON payloads
            json_payloads = self._create_json_payload(control_message, df)

            # Push to broker
            self._push_to_broker(json_payloads, response_channel)

            self.message_count += 1
            self._logger.debug(f"Successfully processed message {control_message.get_metadata('job_id', 'unknown')}")

            return True

        except Exception as e:
            # Handle failure
            self._handle_failure("error_queue", [], e, 0)
            return False

    def start(self):
        """Start the sink."""
        self._running = True
        self.start_time = datetime.now()
        self._logger.info("PythonMessageBrokerTaskSink started")

    def stop(self):
        """Stop the sink."""
        self._running = False
        self._logger.info("PythonMessageBrokerTaskSink stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the sink."""
        return {
            "message_count": self.message_count,
            "running": self._running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }
