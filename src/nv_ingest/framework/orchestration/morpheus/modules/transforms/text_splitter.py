# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import mrc
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.framework.orchestration.morpheus.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager

logger = logging.getLogger(__name__)

MODULE_NAME = "text_splitter"
MODULE_NAMESPACE = "nv_ingest"

TextSplitterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, TextSplitterSchema)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _text_splitter(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """
    validated_config = fetch_and_validate_module_config(builder, TextSplitterSchema)

    @filter_by_task(["split"])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def on_data(message: IngestControlMessage):
        df_payload = message.payload()
        task_config = remove_task_by_type(message, "split")

        df_updated = transform_text_split_and_tokenize_internal(
            df_transform_ledger=df_payload,
            task_config=task_config,
            transform_config=validated_config,
            execution_trace_log=None,
        )

        message.payload(df_updated)

        return message

    split_node = builder.make_node("split_text_on_data", ops.map(on_data))
    builder.register_module_input("input", split_node)
    builder.register_module_output("output", split_node)
