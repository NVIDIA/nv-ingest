# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import mrc
import pandas as pd
from minio import Minio
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from morpheus_llm.service.vdb.milvus_client import DATA_TYPE_MAP
from morpheus_llm.service.vdb.utils import VectorDBServiceFactory
from morpheus_llm.service.vdb.vector_db_service import VectorDBService
from mrc.core import operators as ops
from pymilvus import BulkInsertState
from pymilvus import connections
from pymilvus import utility

from nv_ingest.framework.schemas.framework_vdb_task_sink_schema import VdbTaskSinkSchema
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.orchestration.morpheus.util.modules.config_validator import (
    fetch_and_validate_module_config,
)
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type

logger = logging.getLogger(__name__)

MODULE_NAME = "vdb_task_sink"
MODULE_NAMESPACE = "nv_ingest"

VDBTaskSinkLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, VdbTaskSinkSchema)

_DEFAULT_ENDPOINT = os.environ.get("MINIO_INTERNAL_ADDRESS", "minio:9000")
_DEFAULT_BUCKET_NAME = os.environ.get("MINIO_BUCKET", "nv-ingest")


def _bulk_ingest(
    milvus_uri: str = None,
    collection_name: str = None,
    bucket_name: str = None,
    bulk_ingest_path: str = None,
    extra_params: dict = None,
):
    endpoint = extra_params.get("endpoint", _DEFAULT_ENDPOINT)
    access_key = extra_params.get("access_key", None)
    secret_key = extra_params.get("secret_key", None)

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        session_token=extra_params.get("session_token", None),
        secure=extra_params.get("secure", False),
        region=extra_params.get("region", None),
    )
    bucket_found = client.bucket_exists(bucket_name)
    if not bucket_found:
        raise ValueError(f"Could not find bucket {bucket_name}")
    batch_files = [
        [f"{file.object_name}"] for file in client.list_objects(bucket_name, prefix=bulk_ingest_path, recursive=True)
    ]

    uri_parsed = urlparse(milvus_uri)
    _ = connections.connect(host=uri_parsed.hostname, port=uri_parsed.port)

    task_ids = []
    for file in batch_files:
        task_id = utility.do_bulk_insert(collection_name=collection_name, files=file)
        task_ids.append(task_id)

    while len(task_ids) > 0:
        logger.debug("Wait 1 second to check bulkinsert tasks state...")
        time.sleep(1)
        for id in task_ids:
            state = utility.get_bulk_insert_state(task_id=id)
            if state.state == BulkInsertState.ImportFailed or state.state == BulkInsertState.ImportFailedAndCleaned:
                logger.error(f"The task {state.task_id} failed, reason: {state.failed_reason}")
                task_ids.remove(id)
            elif state.state == BulkInsertState.ImportCompleted:
                logger.debug(f"The task {state.task_id} completed")
                task_ids.remove(id)

    while True:
        progress = utility.index_building_progress(collection_name)
        logger.info(progress)
        if progress.get("total_rows") == progress.get("indexed_rows"):
            break
        time.sleep(5)


def _preprocess_vdb_resources(service, recreate: bool, resource_schemas: dict):
    for resource_name, resource_schema_config in resource_schemas.items():
        has_object = service.has_store_object(name=resource_name)

        if recreate and has_object:
            # Delete the existing resource
            service.drop(name=resource_name)
            has_object = False

        # Ensure that the resource exists
        if not has_object:
            # TODO(Devin)
            import pymilvus

            schema_fields = []
            for field_data in resource_schema_config["schema_conf"]["schema_fields"]:
                if "dtype" in field_data:
                    field_data["dtype"] = DATA_TYPE_MAP.get(field_data["dtype"])
                    field_schema = pymilvus.FieldSchema(**field_data)
                    schema_fields.append(field_schema.to_dict())
                else:
                    schema_fields.append(field_data)

            resource_schema_config["schema_conf"]["schema_fields"] = schema_fields
            # function that we need to call first to turn resource_kwargs into a milvus config spec.

            service.create(name=resource_name, **resource_schema_config)


def _create_vdb_service(
    service: str, is_service_serialized: bool, service_kwargs: dict, recreate: bool, resource_schemas: dict
):
    """
    A function to used to instantiate a `VectorDBService` if a running VDB is available and a connection can be
    established.

    Parameters
    ----------
    service : str
        A string mapping to a supported `VectorDBService`.
    is_service_serialized : bool
        A flag to identify if the supplied service is serialized or needs to be instantiated.
    service_kwargs : dict
        Additional parameters needed to connect to the specificed `VectorDBService`.
    recreate : bool
        A flag specifying whether or not to re-instantate the VDB collection.
    resource_schemas : dict
        Defines the schemas of the VDB collection.

    Returns
    -------
    VectorDBService or str
        If a connection is established, a `VectorDBService` instance is returned, otherwise a string representing
        a supported VDB service is returned to allow repeat connection attempts.
    bool
        A flag used to signify the successful instantiation of a `VectorDBService`.
    """

    service_str = service

    try:
        service: VectorDBService = (
            pickle.loads(bytes(service, "latin1"))
            if is_service_serialized
            else VectorDBServiceFactory.create_instance(service_name=service, **service_kwargs)
        )
        _preprocess_vdb_resources(service, recreate, resource_schemas)

        return service, True

    except Exception as e:
        logger.error(f"Failed to connect to {service_str}: {e}")

        return service_str, False


@dataclass
class AccumulationStats:
    """
    A data class to store accumulation statistics to support dynamic batching of database inserts.

    Attributes
    ----------
    msg_count : int
        Total number of accumulated records.
    last_insert_time : float
        A value representing the time of the most recent database insert.
    data : list[pd.DataFrame]
        A list containing accumulated batches since the last database insert.

    """

    msg_count: int
    last_insert_time: float
    data: list[pd.DataFrame]


def _extract_dataframe_from_control_message(
    ctrl_msg: IngestControlMessage, filter_errors: bool
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Extracts a DataFrame from the control message and applies filtering to remove error messages.
    Returns a tuple of the processed DataFrame and an optional resource name (always None in this case).
    """
    df_payload = ctrl_msg.payload()

    if filter_errors:
        info_msg_mask = df_payload["metadata"].struct.field("info_message_metadata").struct.field("filter")
        df_payload = df_payload.loc[~info_msg_mask].copy()

    # Extract necessary fields from metadata.
    df_payload["embedding"] = df_payload["metadata"].struct.field("embedding")
    df_payload["_source_metadata"] = df_payload["metadata"].struct.field("source_metadata")
    df_payload["_content_metadata"] = df_payload["metadata"].struct.field("content_metadata")

    # Filter rows that contain embeddings and select columns.
    df = df_payload[df_payload["_contains_embeddings"]].copy()
    df = df[
        [
            "embedding",
            "_content",
            "_source_metadata",
            "_content_metadata",
        ]
    ]
    df.columns = ["vector", "text", "source", "content_metadata"]

    return df, None


def _update_accumulator_and_flush(
    df: pd.DataFrame,
    resource_name: Optional[str],
    accumulator_dict: Dict[str, AccumulationStats],
    batch_size: int,
    write_time_interval: float,
    service: VectorDBService,
    resource_kwargs: dict,
    ctrl_msg: IngestControlMessage,
    default_resource_name: str,
) -> None:
    """
    Updates the accumulator for the given resource with the new DataFrame.
    Flushes (inserts) data if the batch size or time interval criteria are met.
    """
    if resource_name is None:
        resource_name = default_resource_name

    if not service.has_store_object(resource_name):
        logger.error("Resource not exists in the vector database: %s", resource_name)
        raise ValueError(f"Resource not exists in the vector database: {resource_name}")

    if resource_name in accumulator_dict:
        accumulator = accumulator_dict[resource_name]
        accumulator.msg_count += len(df)
        accumulator.data.append(df)
    else:
        accumulator_dict[resource_name] = AccumulationStats(msg_count=len(df), last_insert_time=time.time(), data=[df])

    current_time = time.time()
    for key, accum_stats in accumulator_dict.items():
        if accum_stats.msg_count >= batch_size or (
            accum_stats.last_insert_time != -1 and (current_time - accum_stats.last_insert_time) >= write_time_interval
        ):
            if accum_stats.data:
                merged_df = pd.concat(accum_stats.data, ignore_index=True)
                service.insert_dataframe(name=key, df=merged_df, **resource_kwargs)
                accum_stats.data.clear()
                accum_stats.last_insert_time = current_time
                accum_stats.msg_count = 0
                ctrl_msg.set_metadata(
                    "insert_response",
                    {
                        "status": "inserted",
                        "accum_count": 0,
                        "insert_count": len(df),
                        "succ_count": len(df),
                        "err_count": 0,
                    },
                )
        else:
            logger.debug("Accumulated %d rows for collection: %s", accum_stats.msg_count, key)
            ctrl_msg.set_metadata(
                "insert_response",
                {
                    "status": "accumulated",
                    "accum_count": len(df),
                    "insert_count": 0,
                    "succ_count": 0,
                    "err_count": 0,
                },
            )


def _finalize_vector_db_service(accumulator_dict: Dict[str, AccumulationStats], service: VectorDBService) -> None:
    """
    Flushes any remaining accumulated data to the vector database and closes the service connection.
    """
    for key, accum_stats in accumulator_dict.items():
        try:
            if accum_stats.data:
                merged_df = pd.concat(accum_stats.data, ignore_index=True)
                service.insert_dataframe(name=key, df=merged_df)
        except Exception as e:
            logger.error("Unable to upload dataframe entries to vector database: %s", e)
    if isinstance(service, VectorDBService):
        service.close()


def _process_control_message_data(
    ctrl_msg: IngestControlMessage,
    service: VectorDBService,
    accumulator_dict: Dict[str, AccumulationStats],
    default_resource_name: str,
    batch_size: int,
    write_time_interval: float,
    resource_kwargs: dict,
    service_kwargs: dict,
    filter_errors: bool,
) -> IngestControlMessage:
    """
    Processes the control message for data ingestion. If bulk ingestion is enabled, delegates to the bulk ingest
    function.
    Otherwise, it extracts the DataFrame from the message and updates/flushed the accumulator accordingly.
    """
    task_props = remove_task_by_type(ctrl_msg, "vdb_upload")
    bulk_ingest = task_props.get("bulk_ingest", False)
    bulk_ingest_path = task_props.get("bulk_ingest_path", None)
    bucket_name = task_props.get("bucket_name", _DEFAULT_BUCKET_NAME)
    extra_params = task_props.get("params", {})
    filter_errors = task_props.get("filter_errors", filter_errors)

    if bulk_ingest:
        _bulk_ingest(service_kwargs["uri"], default_resource_name, bucket_name, bulk_ingest_path, extra_params)
        return ctrl_msg
    else:
        df, msg_resource_target = _extract_dataframe_from_control_message(ctrl_msg, filter_errors)
        if df is not None and not df.empty:
            # Ensure that df is a pandas DataFrame.
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            _update_accumulator_and_flush(
                df,
                msg_resource_target,
                accumulator_dict,
                batch_size,
                write_time_interval,
                service,
                resource_kwargs,
                ctrl_msg,
                default_resource_name,
            )
        return ctrl_msg


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _vdb_task_sink(builder: mrc.Builder):
    """
    Receives incoming messages in IngestControlMessage format and writes data to a vector database.

    The module configuration (validated using VdbTaskSinkSchema) should include various parameters
    controlling resource creation, batching, write intervals, and retry intervals.
    """
    validated_config = fetch_and_validate_module_config(builder, VdbTaskSinkSchema)
    recreate = validated_config.recreate
    service = validated_config.service
    is_service_serialized = validated_config.is_service_serialized
    default_resource_name = validated_config.default_resource_name
    resource_kwargs = validated_config.resource_kwargs
    resource_schemas = validated_config.resource_schemas
    service_kwargs = validated_config.service_kwargs
    batch_size = validated_config.batch_size
    write_time_interval = validated_config.write_time_interval
    retry_interval = validated_config.retry_interval
    start_time = time.time()

    service, service_status = _create_vdb_service(
        service, is_service_serialized, service_kwargs, recreate, resource_schemas
    )

    accumulator_dict = {default_resource_name: AccumulationStats(msg_count=0, last_insert_time=time.time(), data=[])}

    # on_completed callback
    def on_completed():
        _finalize_vector_db_service(accumulator_dict, service)

    @filter_by_task(["vdb_upload"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: IngestControlMessage):
        nonlocal service_status, start_time, service

        try:
            task_props = remove_task_by_type(ctrl_msg, "vdb_upload")
            bulk_ingest = task_props.get("bulk_ingest", False)
            _ = bulk_ingest

            # Reconnect service if necessary.
            if not service_status:
                curr_time = time.time()
                if curr_time - start_time >= retry_interval:
                    service, service_status = _create_vdb_service(
                        service, is_service_serialized, service_kwargs, recreate, resource_schemas
                    )
                    start_time = curr_time

            if not service_status:
                logger.error("Not connected to vector database.")
                raise ValueError("Not connected to vector database")

            ctrl_msg = _process_control_message_data(
                ctrl_msg,
                service,
                accumulator_dict,
                default_resource_name,
                batch_size,
                write_time_interval,
                resource_kwargs,
                service_kwargs,
                filter_errors=True,
            )

        except Exception as e:
            raise ValueError(f"Failed to insert upload to vector database: {e}")

        return ctrl_msg

    node = builder.make_node(
        WRITE_TO_VECTOR_DB,
        ops.map(on_data),
        ops.filter(lambda val: val is not None),
        ops.on_completed(on_completed),
    )
    node.launch_options.engines_per_pe = validated_config.progress_engines

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
