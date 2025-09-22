# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import json
import logging
from typing import Any, Dict, List, Tuple, Literal, Optional, Union
from pydantic import BaseModel, Field
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient

from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.data_handlers.data_writer import (
    IngestDataWriter,
    RedisDestinationConfig,
    FilesystemDestinationConfig,
)

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


# --- Update the Main Sink Configuration ---


class MessageBrokerTaskSinkConfig(BaseModel):
    """
    Configuration for the MessageBrokerTaskSinkStage.

    Attributes
    ----------
    broker_client : Union[RedisClientConfig, SimpleClientConfig]
        Configuration parameters for connecting to the message broker.
        The specific schema is determined by the 'client_type' field.
    poll_interval : float, optional
        The polling interval (in seconds) for processing messages. Defaults to 0.1.
    """

    # Use the discriminated union for broker_client
    broker_client: Union[RedisClientConfig, SimpleClientConfig] = Field(..., discriminator="client_type")
    poll_interval: float = Field(default=0.1, gt=0)


@ray.remote
class MessageBrokerTaskSinkStage(RayActorStage):
    def __init__(self, config: MessageBrokerTaskSinkConfig, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=False, stage_name=stage_name)

        self.config: MessageBrokerTaskSinkConfig
        self.poll_interval = self.config.poll_interval

        # Create the appropriate broker client (e.g., Redis or Simple).
        self.client = self._create_client()

        # NEW: Create the data writer for external system outputs
        self.data_writer = IngestDataWriter(max_workers=4)

        self.start_time = None
        self.message_count = 0

    # --- Private Helper Methods ---
    def _create_client(self):
        broker_config = self.config.broker_client

        if broker_config.client_type == "redis":
            return RedisClient(
                host=broker_config.host,
                port=broker_config.port,
                db=broker_config.broker_params.db,  # Access nested Pydantic model
                max_retries=broker_config.max_retries,
                max_backoff=broker_config.max_backoff,
                connection_timeout=broker_config.connection_timeout,
                use_ssl=broker_config.broker_params.use_ssl,  # Access nested Pydantic model
            )
        elif broker_config.client_type == "simple":
            server_host = broker_config.host
            server_host = "0.0.0.0"
            return SimpleClient(
                host=server_host,  # Using the potentially overridden host
                port=broker_config.port,
                max_retries=broker_config.max_retries,
                max_backoff=broker_config.max_backoff,
                connection_timeout=broker_config.connection_timeout,
                # broker_params is available via broker_config.broker_params if needed
            )

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

    def _execute_output(
        self, output_destination: str, json_result_fragments: List[Dict[str, Any]], response_channel: str
    ) -> None:
        """
        Execute output using the IngestDataWriter.

        Parameters
        ----------
        output_destination : str
            Where to send output ('redis' or filesystem URI)
        json_result_fragments : List[Dict[str, Any]]
            JSON payload fragments to output
        response_channel : str
            Broker channel or base path for filesystem output
        """
        # Convert fragments to JSON strings
        json_payloads = [json.dumps(fragment) for fragment in json_result_fragments]

        # Calculate and log sizes (unchanged logic)
        total_payload_size = sum(len(payload.encode("utf-8")) for payload in json_payloads)
        total_size_mb = total_payload_size / (1024 * 1024)
        logger.debug(f"Sink Total JSON payload size: {total_size_mb:.2f} MB")

        # Create destination config and write asynchronously
        if output_destination == "redis":
            # Use existing broker config for Redis
            broker_config = self.config.broker_client
            if hasattr(broker_config, "host") and hasattr(broker_config, "port"):
                dest_config = RedisDestinationConfig(
                    host=broker_config.host,
                    port=getattr(broker_config, "port", 6379),
                    db=getattr(broker_config, "db", 0),
                    password=getattr(broker_config, "password", None),
                    channel=response_channel,
                )
            else:
                # Fallback: assume SimpleClient, but for Redis we need proper config
                raise ValueError("Redis output requires Redis broker configuration")
        else:
            # Filesystem destination
            dest_path = self._resolve_destination_path(output_destination, response_channel)
            dest_config = FilesystemDestinationConfig(path=dest_path)

        # Write asynchronously using the data writer
        self.data_writer.write_async(json_payloads, dest_config)

    def _handle_async_output_result(self, future) -> None:
        """
        Handle completion of asynchronous output tasks.

        Parameters
        ----------
        future : Future
            The completed future from the async output operation
        """
        try:
            # Check if the task completed successfully
            future.result()  # This will raise an exception if the task failed
        except Exception as e:
            logger.error(f"Asynchronous output operation failed: {e}", exc_info=True)

    def _resolve_destination_path(self, destination: str, response_channel: str) -> str:
        """
        Resolve the full destination path for filesystem output.

        Parameters
        ----------
        destination : str
            Base destination URI
        response_channel : str
            Response channel (used for filename if destination is a directory)

        Returns
        -------
        str
            Full destination path
        """
        # If destination ends with '/', treat as directory and append filename
        if destination.endswith("/"):
            return f"{destination.rstrip('/')}/{response_channel}.json"
        # Otherwise, use destination directly
        return destination

    # --- Public API Methods for message broker sink ---

    @udf_intercept_hook()
    def on_data(self, control_message: Any) -> Any:
        """
        Process control message through declarative phases:
        1. Extract metadata
        2. Determine processing path (success/failure)
        3. Extract and transform data
        4. Resolve output destination
        5. Create output payloads
        6. Execute output
        7. Handle completion
        """
        # Phase 1: Extract metadata
        response_channel = control_message.get_metadata("response_channel")
        output_destination = control_message.get_metadata("output_destination", "redis")
        cm_failed = control_message.get_metadata("cm_failed", False)

        # Phase 2: Determine processing path
        try:
            # Phase 3: Extract and transform data
            if not cm_failed:
                mdf, df_json = self._extract_data_frame(control_message)
            else:
                mdf, df_json = None, None

            # Phase 4: Create output payloads (unchanged logic)
            json_result_fragments = self._create_json_payload(control_message, df_json)

            # Phase 5: Execute output based on destination
            self._execute_output(output_destination, json_result_fragments, response_channel)

            # Phase 6: Handle success
            annotate_cm(control_message, message="Pushed")

        except ValueError as e:
            # Phase 7: Handle failure
            mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
            self._handle_failure(response_channel, json_result_fragments, e, mdf_size)
        except Exception as e:
            logger.exception(f"Critical error processing message: {e}")
            mdf_size = len(mdf) if mdf is not None and not mdf.empty else 0
            self._handle_failure(response_channel, json_result_fragments, e, mdf_size)

        self.message_count += 1
        self._logger.debug(f"[Message Broker Sink] Processed message count: {self.message_count}")

        return control_message
