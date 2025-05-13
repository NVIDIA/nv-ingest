# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from typing import Dict

import mrc
import mrc.core.operators as ops
import pandas as pd
from morpheus.utils.module_utils import register_module, ModuleLoaderFactory

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.store.image_upload import store_images_to_minio_internal
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest.framework.orchestration.morpheus.util.modules.config_validator import (
    fetch_and_validate_module_config,
)
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.primitives.ingest_control_message import remove_task_by_type, IngestControlMessage

logger = logging.getLogger(__name__)

MODULE_NAME = "image_storage"
MODULE_NAMESPACE = "nv_ingest"

ImageStorageLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, ImageStorageModuleSchema)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _storage_images(builder: mrc.Builder) -> None:
    """
    A module for storing images or structured content in MinIO, updating the metadata
    with storage URLs.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus pipeline builder object.

    Raises
    ------
    ValueError
        If storing extracted objects fails.
    """
    validated_config = fetch_and_validate_module_config(builder, ImageStorageModuleSchema)

    @filter_by_task(["store"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(ctrl_msg: IngestControlMessage) -> IngestControlMessage:
        try:
            task_config = remove_task_by_type(ctrl_msg, "store")

            store_structured: bool = task_config.get("structured", True)
            store_unstructured: bool = task_config.get("images", False)

            content_types: Dict[Any, Any] = {}
            if store_structured:
                content_types[ContentTypeEnum.STRUCTURED] = store_structured
            if store_unstructured:
                content_types[ContentTypeEnum.IMAGE] = store_unstructured

            params: Dict[str, Any] = task_config.get("params", {})
            params["content_types"] = content_types

            logger.debug(f"Processing storage task with parameters: {params}")

            df_storage_ledger = ctrl_msg.payload()

            # Delegate all DataFrame-related work to our free function.
            df_storage_ledger: pd.DataFrame = store_images_to_minio_internal(
                df_storage_ledger=df_storage_ledger, task_config=params, storage_config={}, execution_trace_log=None
            )

            # Update the control message payload with the new DataFrame.
            ctrl_msg.payload(df_storage_ledger)
        except Exception as e:
            err_msg = f"Failed to store extracted objects: {e}"
            logger.exception(err_msg)

            raise type(e)(err_msg)

        return ctrl_msg

    input_node = builder.make_node("image_storage", ops.map(on_data))
    builder.register_module_input("input", input_node)
    builder.register_module_output("output", input_node)
