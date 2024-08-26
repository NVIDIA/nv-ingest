# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import mrc
from morpheus.messages import ControlMessage
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

from nv_ingest.schemas.task_injection_schema import TaskInjectionSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "task_injection"
MODULE_NAMESPACE = "nv_ingest"

TaskInjectorLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, TaskInjectionSchema)


def on_data(message: ControlMessage):
    message.get_metadata("task_meta")

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _task_injection(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, TaskInjectionSchema)

    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    @traceable(MODULE_NAME)
    def _on_data(ctrl_msg: ControlMessage):
        return on_data(ctrl_msg)
        ctrl_msg.get_metadata("task_meta")

        return ctrl_msg

    node = builder.make_node("vdb_resource_tagging", on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
