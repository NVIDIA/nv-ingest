# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import pickle
import time
from dataclasses import dataclass

import mrc
from morpheus.messages import ControlMessage
from morpheus_llm.service.vdb.milvus_client import DATA_TYPE_MAP
from morpheus_llm.service.vdb.utils import VectorDBServiceFactory
from morpheus_llm.service.vdb.vector_db_service import VectorDBService
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_ids import WRITE_TO_VECTOR_DB
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

import cudf

from nv_ingest.schemas.vdb_task_sink_schema import VdbTaskSinkSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "vdb_task_sink"
MODULE_NAMESPACE = "nv_ingest"

VDBTaskSinkLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, VdbTaskSinkSchema)


def preprocess_vdb_resources(service, recreate: bool, resource_schemas: dict):
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
        preprocess_vdb_resources(service, recreate, resource_schemas)

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
    data : list[cudf.DataFrame]
        A list containing accumulated batches since the last database insert.

    """

    msg_count: int
    last_insert_time: float
    data: list[cudf.DataFrame]


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _vdb_task_sink(builder: mrc.Builder):
    """
    Receives incoming messages in ControlMessage format.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    Notes
    -----
    The `module_config` should contain:
    - 'recreate': bool, whether to recreate the resource if it already exists (default is False).
    - 'service': str, the name of the service or a serialized instance of VectorDBService.
    - 'is_service_serialized': bool, whether the provided service is serialized (default is False).
    - 'default_resource_name': str, the name of the collection resource (must not be None or empty).
    - 'resource_kwargs': dict, additional keyword arguments for resource creation.
    - 'resource_schemas': dict, additional keyword arguments for resource creation.
    - 'service_kwargs': dict, additional keyword arguments for VectorDBService creation.
    - 'batch_size': int, accumulates messages until reaching the specified batch size for writing to VDB.
    - 'write_time_interval': float, specifies the time interval (in seconds) for writing messages, or writing messages
    - 'retry_interval': float, specify the interval to retry connections to milvus
    when the accumulated batch size is reached.

    Raises
    ------
    ValueError
        If 'resource_name' is None or empty.
        If 'service' is not provided or is not a valid service name or a serialized instance of VectorDBService.
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

    def on_completed():
        final_df_references = []

        # Pushing remaining messages
        for key, accum_stats in accumulator_dict.items():
            try:
                if accum_stats.data:
                    merged_df = cudf.concat(accum_stats.data)
                    service.insert_dataframe(name=key, df=merged_df)
                    final_df_references.append(accum_stats.data)
            except Exception as e:
                logger.error("Unable to upload dataframe entries to vector database: %s", e)
        # Close vector database service connection
        if isinstance(service, VectorDBService):
            service.close()

    def extract_df(ctrl_msg: ControlMessage, filter_errors: bool):
        df = None
        resource_name = None

        with ctrl_msg.payload().mutable_dataframe() as mdf:
            # info_msg mask
            if filter_errors:
                info_msg_mask = mdf["metadata"].struct.field("info_message_metadata").struct.field("filter")
                mdf = mdf.loc[~info_msg_mask].copy()

            mdf["embedding"] = mdf["metadata"].struct.field("embedding")
            mdf["_source_metadata"] = mdf["metadata"].struct.field("source_metadata")
            df = mdf[mdf["_contains_embeddings"]].copy()

        df = df[
            [
                "embedding",
                "_content",
                "_source_metadata",
            ]
        ]
        df.columns = ["vector", "text", "source"]

        return df, resource_name

    @filter_by_task(["vdb_upload"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: ControlMessage):
        nonlocal service_status
        nonlocal start_time
        nonlocal service

        try:
            task_props = ctrl_msg.remove_task("vdb_upload")
            filter_errors = task_props.get("filter_errors", True)

            if not service_status:
                curr_time = time.time()
                delta_t = curr_time - start_time
                if delta_t >= retry_interval:
                    service, service_status = _create_vdb_service(
                        service, is_service_serialized, service_kwargs, recreate, resource_schemas
                    )
                    start_time = curr_time

            if not service_status:
                logger.error("Not connected to vector database.")
                raise ValueError("Not connected to vector database")

            df, msg_resource_target = extract_df(ctrl_msg, filter_errors)

            if df is not None and not df.empty:
                if not isinstance(df, cudf.DataFrame):
                    df = cudf.DataFrame(df)

                df_size = len(df)
                current_time = time.time()

                # Use default resource name
                if not msg_resource_target:
                    msg_resource_target = default_resource_name
                    if not service.has_store_object(msg_resource_target):
                        logger.error("Resource not exists in the vector database: %s", msg_resource_target)
                        raise ValueError(f"Resource not exists in the vector database: {msg_resource_target}")

                if msg_resource_target in accumulator_dict:
                    accumulator: AccumulationStats = accumulator_dict[msg_resource_target]
                    accumulator.msg_count += df_size
                    accumulator.data.append(df)
                else:
                    accumulator_dict[msg_resource_target] = AccumulationStats(
                        msg_count=df_size, last_insert_time=time.time(), data=[df]
                    )

                for key, accum_stats in accumulator_dict.items():
                    if accum_stats.msg_count >= batch_size or (
                        accum_stats.last_insert_time != -1
                        and (current_time - accum_stats.last_insert_time) >= write_time_interval
                    ):
                        if accum_stats.data:
                            merged_df = cudf.concat(accum_stats.data)

                            # pylint: disable=not-a-mapping
                            service.insert_dataframe(name=key, df=merged_df, **resource_kwargs)
                            # Reset accumulator stats
                            accum_stats.data.clear()
                            accum_stats.last_insert_time = current_time
                            accum_stats.msg_count = 0

                        if isinstance(ctrl_msg, ControlMessage):
                            ctrl_msg.set_metadata(
                                "insert_response",
                                {
                                    "status": "inserted",
                                    "accum_count": 0,
                                    "insert_count": df_size,
                                    "succ_count": df_size,
                                    "err_count": 0,
                                },
                            )
                    else:
                        logger.debug("Accumulated %d rows for collection: %s", accum_stats.msg_count, key)
                        if isinstance(ctrl_msg, ControlMessage):
                            ctrl_msg.set_metadata(
                                "insert_response",
                                {
                                    "status": "accumulated",
                                    "accum_count": df_size,
                                    "insert_count": 0,
                                    "succ_count": 0,
                                    "err_count": 0,
                                },
                            )

        except Exception as e:
            raise ValueError(f"Failed to insert upload to vector database: {e}")

        return ctrl_msg

    node = builder.make_node(
        WRITE_TO_VECTOR_DB, ops.map(on_data), ops.filter(lambda val: val is not None), ops.on_completed(on_completed)
    )

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
