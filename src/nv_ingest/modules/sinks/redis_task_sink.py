# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import sys
from typing import Any
from typing import Dict
from typing import Tuple

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from redis import RedisError

from nv_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.redis import RedisClient
from nv_ingest.util.tracing import traceable
from nv_ingest.util.tracing.logging import annotate_cm

logger = logging.getLogger(__name__)

MODULE_NAME = "redis_task_sink"
MODULE_NAMESPACE = "nv_ingest"

RedisTaskSinkLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def extract_data_frame(message: ControlMessage) -> Tuple[Any, Dict[str, Any]]:
    """
    Extracts a DataFrame from a message payload and returns it along with a filtered dictionary of required columns.

    Parameters
    ----------
    message : ControlMessage
        The message object containing the payload.

    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        A tuple containing the mutable DataFrame and a dictionary of selected columns.
    """
    try:
        with message.payload().mutable_dataframe() as mdf:
            logger.debug(f"Redis Sink Received DataFrame with {len(mdf)} rows.")
            keep_cols = ["document_type", "metadata"]
            return mdf, mdf[keep_cols].to_dict(orient="records")
    except Exception:
        return None, None


def create_json_payload(message: ControlMessage, df_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a JSON payload based on message status and data, including optional trace and annotation data.

    Parameters
    ----------
    message : ControlMessage
        The message object from which metadata is extracted.
    df_json : Dict[str, Any]
        The dictionary containing data filtered from the DataFrame.

    Returns
    -------
    Dict[str, Any]
        The JSON payload to be forwarded.
    """
    message_status = "success" if not message.get_metadata("cm_failed", False) else "failed"
    description = (
        "Successfully processed the message." if message_status == "success" else "Failed to process the message."
    )
    ret_val_json = {
        "status": message_status,
        "description": description,
        "data": df_json,
    }

    if message.get_metadata("add_trace_tagging", True):
        ret_val_json["trace"] = {
            key: message.get_timestamp(key).timestamp() * 1e9 for key in message.filter_timestamp("trace::")
        }
        ret_val_json["annotations"] = {
            key: message.get_metadata(key) for key in message.list_metadata() if key.startswith("annotation::")
        }

    return ret_val_json


def push_to_redis(redis_client: RedisClient, response_channel: str, json_payload: str, retry_count: int = 2) -> None:
    """
    Attempts to push a JSON payload to a Redis channel, retrying on failure up to a specified number of attempts.

    Parameters
    ----------
    redis_client : RedisClient
        The Redis client used to push the data.
    response_channel : str
        The Redis channel to which the data is pushed.
    json_payload : str
        The JSON string payload to be pushed.
    retry_count : int, optional
        The number of attempts to retry on failure (default is 2).

    Returns
    -------
    None

    Raises
    ------
    RedisError
        If pushing to Redis fails after the specified number of retries.
    """
    payload_size = sys.getsizeof(json_payload)
    size_limit = 2**28  # 256 MB
    if payload_size > size_limit:
        raise RedisError(f"Payload size {payload_size} bytes exceeds limit of {size_limit / 1e6} MB.")

    for attempt in range(retry_count):
        try:
            redis_client.get_client().rpush(response_channel, json_payload)
            logger.debug(f"Redis Sink Forwarded message to Redis channel '{response_channel}'.")
            return
        except RedisError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_count - 1:
                raise


def handle_failure(redis_client, response_channel, ret_val_json, e, mdf_size):
    error_description = (
        f"Failed to forward message to Redis after retries: {e}. "
        f"Payload size: {sys.getsizeof(json.dumps(ret_val_json)) / 1e6} MB, Rows: {mdf_size}"
    )
    logger.error(error_description)

    # Construct a failure message and push it
    fail_msg = {
        "data": None,
        "status": "failed",
        "description": error_description,
        "trace": ret_val_json.get("trace", {}),
    }
    redis_client.get_client().rpush(response_channel, json.dumps(fail_msg))


def process_and_forward(message: ControlMessage, redis_client: RedisClient) -> ControlMessage:
    """
    Processes a message by extracting data, creating a JSON payload, and attempting to push it to Redis.

    Parameters
    ----------
    message : ControlMessage
        The message to process.
    redis_client : RedisClient
        The Redis client used for pushing data.

    Returns
    -------
    ControlMessage
        The processed message.

    Raises
    ------
    Exception
        If a critical error occurs during processing.
    """
    try:
        cm_failed = message.get_metadata("cm_failed", False)
        if not cm_failed:
            mdf, df_json = extract_data_frame(message)
            ret_val_json = create_json_payload(message, df_json)
        else:
            ret_val_json = create_json_payload(message, None)

        json_payload = json.dumps(ret_val_json)
        annotate_cm(message, message="Pushed")
        response_channel = message.get_metadata("response_channel")
        push_to_redis(redis_client, response_channel, json_payload)
    except RedisError as e:
        mdf_size = len(mdf) if mdf else 0
        handle_failure(redis_client, response_channel, ret_val_json, e, mdf_size)
    except Exception as e:
        logger.error(f"Critical error processing message: {e}")

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _redis_task_sink(builder: mrc.Builder) -> None:
    """
    Configures and registers a processing node for message handling, including Redis task sinking within a modular
    processing chain. This function initializes a Redis client based on provided configuration, wraps the
    `process_and_forward` function for message processing, and sets up a processing node.

    Parameters
    ----------
    builder : mrc.Builder
        The modular processing chain builder to which the Redis task sink node will be added.

    Returns
    -------
    None

    Notes
    -----
    This setup applies necessary decorators for failure handling and trace tagging. The node is then registered as
    both an input and an output module in the builder, completing the setup for message processing and
    forwarding to Redis. It ensures that all messages passed through this node are processed and forwarded
    efficiently with robust error handling and connection management to Redis.
    """
    validated_config = fetch_and_validate_module_config(builder, RedisTaskSinkSchema)

    # Initialize RedisClient with the validated configuration
    redis_client = RedisClient(
        host=validated_config.redis_client.host,
        port=validated_config.redis_client.port,
        db=0,  # Assuming DB is always 0 for simplicity; make configurable if needed
        max_retries=validated_config.redis_client.max_retries,
        max_backoff=validated_config.redis_client.max_backoff,
        connection_timeout=validated_config.redis_client.connection_timeout,
        use_ssl=validated_config.redis_client.use_ssl,
    )

    @traceable(MODULE_NAME)
    def _process_and_forward(message: ControlMessage) -> ControlMessage:
        """
        Wraps the processing and forwarding functionality with traceability and error handling.

        Parameters
        ----------
        message : ControlMessage
            The message to be processed and forwarded to Redis.

        Returns
        -------
        ControlMessage
            The processed message, after attempting to forward to Redis.
        """
        return process_and_forward(message, redis_client)

    process_node = builder.make_node("process_and_forward", ops.map(_process_and_forward))
    process_node.launch_options.engines_per_pe = validated_config.progress_engines

    # Register the final output of the module
    builder.register_module_input("input", process_node)
    builder.register_module_output("output", process_node)
