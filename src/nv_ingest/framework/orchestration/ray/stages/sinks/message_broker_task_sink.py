# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import json
import threading
import logging
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient, SimpleMessageBroker
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

logger = logging.getLogger(__name__)


class MessageBrokerTaskSinkConfig(BaseModel):
    """
    Configuration for the MessageBrokerTaskSinkStage.

    Attributes
    ----------
    broker_client : dict
        Configuration parameters for connecting to the message broker.
    poll_interval : float, optional
        The polling interval (in seconds) for processing messages.
    """

    broker_client: dict
    poll_interval: float = 0.1


@ray.remote
class MessageBrokerTaskSinkStage(RayActorSinkStage):
    def __init__(self, config: BaseModel, progress_engine_count: int) -> None:
        super().__init__(config, progress_engine_count)
        # Configuration specific to the message broker sink.
        self.broker_client = self.config.broker_client
        self.poll_interval = self.config.poll_interval
        # Create the appropriate broker client (e.g., Redis or Simple).
        self.client = self._create_client()
        self.start_time = None
        self.message_count = 0

    # --- Private Helper Methods ---
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
    def _extract_data_frame(message: Any) -> Tuple[Any, Any]:
        """
        Extracts a DataFrame from a message payload and returns it along with selected columns.
        """
        try:
            df = message.payload()
            logger.debug(f"Sink received DataFrame with {len(df)} rows.")
            keep_cols = ["document_type", "metadata"]
            return df, df[keep_cols].to_dict(orient="records")
        except Exception as err:
            logger.warning(f"Failed to extract DataFrame: {err}")
            return None, None

    @staticmethod
    def _split_large_dict(json_data: List[Dict[str, Any]], size_limit: int) -> List[List[Dict[str, Any]]]:
        fragments = []
        current_fragment = []
        current_size = sys.getsizeof(json.dumps(current_fragment))
        for item in json_data:
            item_size = sys.getsizeof(json.dumps(item))
            if current_size + item_size > size_limit:
                fragments.append(current_fragment)
                current_fragment = []
                current_size = sys.getsizeof(json.dumps(current_fragment))
            current_fragment.append(item)
            current_size += item_size
        if current_fragment:
            fragments.append(current_fragment)
        return fragments

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
            if i == 0 and message.get_metadata("add_trace_tagging", True):
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
        for payload in json_payloads:
            payload_size = sys.getsizeof(payload)
            size_limit = 2**28  # 256 MB
            if payload_size > size_limit:
                raise ValueError(f"Payload size {payload_size} exceeds limit of {size_limit / 1e6} MB.")
        for attempt in range(retry_count):
            try:
                for payload in json_payloads:
                    self.client.submit_message(response_channel, payload)
                logger.debug(f"Sink forwarded message to channel '{response_channel}'.")
                return
            except ValueError as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    raise

    def _handle_failure(
        self, response_channel: str, json_result_fragments: List[Dict[str, Any]], e: Exception, mdf_size: int
    ) -> None:
        """
        Handles failure by logging and pushing a failure message to the broker.
        """
        error_description = (
            f"Failed to forward message: {e}. "
            f"Payload size: {sys.getsizeof(json.dumps(json_result_fragments)) / 1e6} MB, "
            f"Rows: {mdf_size}"
        )
        logger.error(error_description)
        fail_msg = {
            "data": None,
            "status": "failed",
            "description": error_description,
            "trace": json_result_fragments[0].get("trace", {}) if json_result_fragments else {},
        }
        self.client.submit_message(response_channel, json.dumps(fail_msg))

    # --- Public API Methods for Sink Stage ---
    async def on_data(self, control_message: Any) -> Any:
        """
        For this sink stage, no additional transformation is performed.
        """

        return control_message

    async def write_output(self, control_message: Any) -> Any:
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
        return control_message
