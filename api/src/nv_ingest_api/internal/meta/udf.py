# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.internal.schemas.meta.udf import UDFStageSchema
from nv_ingest_api.util.imports.callable_signatures import ingest_callable_signature

logger = logging.getLogger(__name__)


def udf_stage_callable_fn(control_message: IngestControlMessage, stage_config: UDFStageSchema) -> IngestControlMessage:
    """
    Process an incoming IngestControlMessage by executing a user-defined function.

    This function:
    1. Extracts the 'udf' task from the control message
    2. Gets the UDF function string from the task or stage config
    3. Evaluates the function string to create a callable
    4. Validates the function signature
    5. Applies the function to the control message
    6. Returns the result

    Parameters
    ----------
    control_message : IngestControlMessage
        The incoming message to process
    stage_config : UDFStageSchema
        Configuration containing the UDF function string

    Returns
    -------
    IngestControlMessage
        The processed control message

    Raises
    ------
    ValueError
        If no UDF function is provided or if the function signature is invalid
    RuntimeError
        If the UDF function execution fails
    """
    logger.debug("Processing UDF stage")

    # Extract 'udf' task from control message
    task_config = remove_task_by_type(control_message, "udf")

    # Get UDF function string from task config
    udf_function_str = task_config.get("udf_function") if task_config else None

    if not udf_function_str:
        if stage_config.ignore_empty_udf:
            logger.debug(
                "No UDF function provided in task config, but ignore_empty_udf is True. Returning message unchanged."
            )
            return control_message
        else:
            error_msg = (
                "No UDF function provided in task config. UDF tasks must include a 'udf_function' field, or "
                "set ignore_empty_udf=True in stage config."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    logger.debug(f"Evaluating UDF function: {udf_function_str[:100]}...")

    # Create a controlled namespace for the UDF execution
    namespace = {
        "IngestControlMessage": IngestControlMessage,
        "__builtins__": __builtins__,
    }

    try:
        # Evaluate the function string to create a callable
        exec(udf_function_str, namespace)

        # Find the function in the namespace (assume it's the first callable that's not a built-in)
        udf_function = None
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith("__") and name != "IngestControlMessage":
                udf_function = obj
                break

        if udf_function is None:
            raise ValueError("No callable function found in the UDF string")

        # Validate the function signature
        sig = inspect.signature(udf_function)
        ingest_callable_signature(sig)

        logger.debug(f"Executing UDF function: {udf_function.__name__}")

        # Execute the UDF function
        result = udf_function(control_message)

        if not isinstance(result, IngestControlMessage):
            raise RuntimeError(f"UDF function must return IngestControlMessage, got {type(result)}")

        logger.debug("UDF stage processing completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error executing UDF function: {str(e)}")
        raise RuntimeError(f"UDF execution failed: {str(e)}") from e
