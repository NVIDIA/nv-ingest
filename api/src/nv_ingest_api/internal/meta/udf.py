# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
from typing import Any, Dict

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_all_tasks_by_type
from nv_ingest_api.internal.schemas.meta.udf import UDFStageSchema
from nv_ingest_api.util.imports.callable_signatures import ingest_callable_signature

logger = logging.getLogger(__name__)


def udf_stage_callable_fn(control_message: IngestControlMessage, stage_config: UDFStageSchema) -> IngestControlMessage:
    """
    UDF stage callable function that processes UDF tasks in a control message.

    This function extracts all UDF tasks from the control message and executes them sequentially.

    Parameters
    ----------
    control_message : IngestControlMessage
        The control message containing UDF tasks to process
    stage_config : UDFStageSchema
        Configuration for the UDF stage

    Returns
    -------
    IngestControlMessage
        The control message after processing all UDF tasks
    """
    logger.debug("Starting UDF stage processing")

    # Extract all UDF tasks from control message using free function
    try:
        all_task_configs = remove_all_tasks_by_type(control_message, "udf")
    except ValueError:
        # No UDF tasks found
        if stage_config.ignore_empty_udf:
            logger.debug("No UDF tasks found, ignoring as configured")
            return control_message
        else:
            raise ValueError("No UDF tasks found in control message")

    # Process each UDF task sequentially
    for task_num, task_config in enumerate(all_task_configs, 1):
        logger.debug(f"Processing UDF task {task_num} of {len(all_task_configs)}")

        # Get UDF function string and function name from task properties
        udf_function_str = task_config.get("udf_function", "").strip()
        udf_function_name = task_config.get("udf_function_name", "").strip()

        # Skip empty UDF functions if configured to ignore them
        if not udf_function_str:
            if stage_config.ignore_empty_udf:
                logger.debug(f"UDF task {task_num} has empty function, skipping as configured")
                continue
            else:
                raise ValueError(f"UDF task {task_num} has empty function string")

        # Validate that function name is provided
        if not udf_function_name:
            raise ValueError(f"UDF task {task_num} missing required 'udf_function_name' property")

        # Execute the UDF function string in a controlled namespace
        namespace: Dict[str, Any] = {}
        try:
            exec(udf_function_str, namespace)
        except Exception as e:
            raise ValueError(f"UDF task {task_num} failed to execute: {str(e)}")

        # Extract the specified function from the namespace
        if udf_function_name in namespace and callable(namespace[udf_function_name]):
            udf_function = namespace[udf_function_name]
        else:
            raise ValueError(
                f"UDF task {task_num}: Specified UDF function '{udf_function_name}' not found or not callable"
            )

        # Validate the UDF function signature
        try:
            ingest_callable_signature(inspect.signature(udf_function))
        except Exception as e:
            raise ValueError(f"UDF task {task_num} has invalid function signature: {str(e)}")

        # Execute the UDF function with the control message
        try:
            control_message = udf_function(control_message)
        except Exception as e:
            raise ValueError(f"UDF task {task_num} execution failed: {str(e)}")

        # Validate that the UDF function returned an IngestControlMessage
        if not isinstance(control_message, IngestControlMessage):
            raise ValueError(f"UDF task {task_num} must return an IngestControlMessage, got {type(control_message)}")

        logger.debug(f"UDF task {task_num} completed successfully")

    logger.debug(f"UDF stage processing completed. Processed {len(all_task_configs)} UDF tasks")
    return control_message
