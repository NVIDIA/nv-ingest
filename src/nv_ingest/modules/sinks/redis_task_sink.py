# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import sys
import traceback
from typing import Any, List
from typing import Dict
from typing import Tuple

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from redis import RedisError

from nv_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from nv_ingest.util.message_brokers.redis.redis_client import RedisClient
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
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
            return mdf, mdf[keep_cols].to_pandas().to_dict(orient="records")
    except Exception as err:
        logger.warning(f"Failed to extract DataFrame from message payload: {err}")
        return None, None


def split_large_dict(json_data: List[Dict[str, Any]], size_limit: int) -> List[List[Dict[str, Any]]]:
    """
    Splits a large list of dictionaries into smaller fragments, each less than the specified size limit (in bytes).

    Parameters
    ----------
    json_data : List[Dict[str, Any]]
        The list of dictionaries to split.
    size_limit : int
        The maximum size in bytes for each fragment.

    Returns
    -------
    List[List[Dict[str, Any]]]
        A list of fragments, each fragment being a list of dictionaries, within the size limit.
    """

    fragments = []
    current_fragment = []
    current_size = sys.getsizeof(json.dumps(current_fragment))

    for item in json_data:
        item_size = sys.getsizeof(json.dumps(item))

        # If adding this item exceeds the size limit, start a new fragment
        if current_size + item_size > size_limit:
            fragments.append(current_fragment)  # Store the current fragment
            current_fragment = []  # Start a new fragment
            current_size = sys.getsizeof(json.dumps(current_fragment))

        # Add the item (dict) to the current fragment
        current_fragment.append(item)
        current_size += item_size

    # Append the last fragment if it has data
    if current_fragment:
        fragments.append(current_fragment)

    return fragments


def create_json_payload(message: ControlMessage, df_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Creates JSON payloads based on message status and data. If the size of df_json exceeds 256 MB, splits it into
    multiple fragments, each less than 256 MB. Adds optional trace and annotation data to the first fragment.

    Parameters
    ----------
    message : ControlMessage
        The message object from which metadata is extracted.
    df_json : Dict[str, Any]
        The dictionary containing data filtered from the DataFrame.

    Returns
    -------
    List[Dict[str, Any]]
        A list of JSON payloads, possibly split into multiple fragments.
    """
    # Convert df_json to a JSON string to check its size
    df_json_str = json.dumps(df_json)
    df_json_size = sys.getsizeof(df_json_str)

    # 256 MB size limit (in bytes)
    size_limit = 256 * 1024 * 1024

    # If df_json is larger than the size limit, split it into chunks
    if df_json_size > size_limit:
        # Split df_json into fragments, ensuring each is a valid JSON object
        data_fragments = split_large_dict(df_json, size_limit)
        fragment_count = len(data_fragments)
    else:
        # No splitting needed, treat the whole thing as one fragment
        data_fragments = [df_json]
        fragment_count = 1

    # Initialize list to store multiple ret_val_json payloads
    ret_val_json_list = []

    # Process each fragment and add necessary metadata
    for i, fragment_data in enumerate(data_fragments):
        ret_val_json = {
            "status": "success" if not message.get_metadata("cm_failed", False) else "failed",
            "description": (
                "Successfully processed the message." if not message.get_metadata("cm_failed",
                                                                                  False) else "Failed to process the message."
            ),
            "data": fragment_data,  # Fragmented data
            "fragment": i,
            "fragment_count": fragment_count,
        }

        # Only add trace tagging and annotations to the first fragment (i.e., fragment=0)
        if i == 0 and message.get_metadata("add_trace_tagging", True):
            ret_val_json["trace"] = {
                key: message.get_timestamp(key).timestamp() * 1e9 for key in message.filter_timestamp("trace::")
            }
            ret_val_json["annotations"] = {
                key: message.get_metadata(key) for key in message.list_metadata() if key.startswith("annotation::")
            }

        ret_val_json_list.append(ret_val_json)

    logger.debug(f"Redis Sink Created {len(ret_val_json_list)} JSON payloads.")

    return ret_val_json_list


def push_to_redis(redis_client: RedisClient, response_channel: str, json_payloads: List[str],
                  retry_count: int = 2) -> None:
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

    for json_payload in json_payloads:
        payload_size = sys.getsizeof(json_payload)
        size_limit = 2 ** 28  # 256 MB

        if payload_size > size_limit:
            raise RedisError(f"Payload size {payload_size} bytes exceeds limit of {size_limit / 1e6} MB.")

    for attempt in range(retry_count):
        try:
            for json_payload in json_payloads:
                redis_client.get_client().rpush(response_channel, json_payload)

            logger.debug(f"Redis Sink Forwarded message to Redis channel '{response_channel}'.")
            return
        except RedisError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_count - 1:
                raise


def handle_failure(
        redis_client: Any,
        response_channel: str,
        json_result_fragments: List[Dict[str, Any]],
        e: Exception,
        mdf_size: int
) -> None:
    """
    Handles failure scenarios by logging the error and pushing a failure message to a Redis channel.

    Parameters
    ----------
    redis_client : Any
        A Redis client instance capable of interacting with Redis.
        It should have a `get_client()` method that returns a client object with an `rpush()` method.
    response_channel : str
        The Redis channel to which the failure message will be sent.
    json_result_fragments : List[Dict[str, Any]]
        A list of JSON result fragments, where each fragment is a dictionary containing the results of the operation.
        The first fragment is used to extract trace data in the failure message.
    e : Exception
        The exception object that triggered the failure.
    mdf_size : int
        The number of rows in the message data frame (mdf) being processed.

    Returns
    -------
    None
        This function does not return any value. It handles the failure by logging the error and sending a message to Redis.

    Notes
    -----
    The failure message includes the error description, the size of the first JSON result fragment in MB,
    and the number of rows in the data being processed. If trace information is available in the first
    fragment of `json_result_fragments`, it is included in the failure message.

    Examples
    --------
    >>> redis_client = SomeRedisClient()
    >>> response_channel = "response_channel_name"
    >>> json_result_fragments = [{"trace": {"event_1": 123456789}}]
    >>> e = Exception("Network failure")
    >>> mdf_size = 1000
    >>> handle_failure(redis_client, response_channel, json_result_fragments, e, mdf_size)
    """
    error_description = (
        f"Failed to forward message to Redis after retries: {e}. "
        f"Payload size: {sys.getsizeof(json.dumps(json_result_fragments)) / 1e6} MB, Rows: {mdf_size}"
    )
    logger.error(error_description)

    # Construct a failure message and push it to Redis
    fail_msg = {
        "data": None,
        "status": "failed",
        "description": error_description,
        "trace": json_result_fragments[0].get("trace", {}),
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
    mdf = None
    json_result_fragments = []
    response_channel = message.get_metadata("response_channel")

    try:
        cm_failed = message.get_metadata("cm_failed", False)
        if not cm_failed:
            mdf, df_json = extract_data_frame(message)
            json_result_fragments = create_json_payload(message, df_json)
        else:
            json_result_fragments = create_json_payload(message, None)

        json_payloads = [json.dumps(fragment) for fragment in json_result_fragments]
        annotate_cm(message, message="Pushed")
        push_to_redis(redis_client, response_channel, json_payloads)
    except RedisError as e:
        mdf_size = len(mdf) if mdf else 0
        handle_failure(redis_client, response_channel, json_result_fragments, e, mdf_size)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Critical error processing message: {e}")

        mdf_size = len(mdf) if mdf else 0
        handle_failure(redis_client, response_channel, json_result_fragments, e, mdf_size)

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
