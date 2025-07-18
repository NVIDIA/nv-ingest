# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
from pydantic import BaseModel, Field, field_validator, ConfigDict

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type
from nv_ingest_api.util.imports.callable_signatures import ingest_callable_signature

logger = logging.getLogger(__name__)


class UDFStageSchema(BaseModel):
    """
    Schema for UDF stage configuration.

    Validates that the UDF function string is provided and non-empty.
    """

    udf_function: str = Field(..., description="Python function as string to be evaluated and executed")

    @field_validator("udf_function")
    @classmethod
    def validate_udf_function(cls, v):
        if not v or not v.strip():
            raise ValueError("udf_function must be a non-empty string")
        return v.strip()

    model_config = ConfigDict(extra="forbid")


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
        The stage configuration containing the UDF function string

    Returns
    -------
    IngestControlMessage
        The processed message after applying the UDF

    Raises
    ------
    ValueError
        If the UDF function string is invalid or has wrong signature
    Exception
        If the UDF function execution fails
    """
    logger.debug("UDF callable processing message")

    # Remove the 'udf' task to get task-specific configuration
    task_config = remove_task_by_type(control_message, "udf")
    logger.debug("Extracted UDF task config: %s", task_config)

    # Get the UDF function string from task config or stage config
    udf_function_str = task_config.get("udf_function") if task_config else None
    if not udf_function_str:
        udf_function_str = stage_config.udf_function

    logger.debug("UDF function string length: %d characters", len(udf_function_str))

    try:
        # Create a controlled namespace for eval
        namespace = {
            "IngestControlMessage": IngestControlMessage,
            "__builtins__": __builtins__,
        }

        # Evaluate the function string to create a callable
        logger.debug("Evaluating UDF function string")
        exec(udf_function_str, namespace)

        # Find the function in the namespace (should be the last defined function)
        # Look for callable objects that aren't built-ins
        user_functions = [
            obj
            for name, obj in namespace.items()
            if callable(obj) and not name.startswith("__") and name != "IngestControlMessage"
        ]

        if not user_functions:
            raise ValueError("No callable function found in UDF string")

        # Use the first (and hopefully only) user-defined function
        udf_function = user_functions[0]
        logger.debug("Found UDF function: %s", udf_function.__name__)

        # Validate the function signature
        sig = inspect.signature(udf_function)
        logger.debug("Validating UDF function signature: %s", sig)
        ingest_callable_signature(sig)

        # Apply the UDF to the control message
        logger.debug("Executing UDF function")
        result = udf_function(control_message)

        if not isinstance(result, IngestControlMessage):
            raise ValueError(f"UDF function must return IngestControlMessage, got {type(result)}")

        logger.info("UDF callable completed successfully")
        return result

    except Exception as e:
        logger.error("UDF execution failed: %s", str(e))
        logger.exception("UDF execution error details")
        raise ValueError(f"UDF execution failed: {str(e)}") from e
