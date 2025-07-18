# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import inspect
from pydantic import BaseModel

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


def ingest_stage_callable_signature(sig: inspect.Signature):
    """
    Validates that a callable has the signature:
        (IngestControlMessage, BaseModel) -> IngestControlMessage

    Also allows for generic (*args, **kwargs) signatures for flexibility with class constructors.

    Raises
    ------
    TypeError
        If the signature does not match the expected pattern.
    """
    params = list(sig.parameters.values())

    # If the signature accepts arbitrary keyword arguments, it's flexible enough.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return

    if len(params) != 2:
        raise TypeError(f"Expected exactly 2 parameters, got {len(params)}")

    if params[0].name != "control_message" or params[1].name != "stage_config":
        raise TypeError("Expected parameter names: 'control_message', 'stage_config'")

    first_param = params[0].annotation
    second_param = params[1].annotation
    return_type = sig.return_annotation

    if first_param is inspect.Parameter.empty:
        raise TypeError("First parameter must be annotated with IngestControlMessage")

    if second_param is inspect.Parameter.empty:
        raise TypeError("Second parameter must be annotated with a subclass of BaseModel")

    if return_type is inspect.Signature.empty:
        raise TypeError("Return type must be annotated with IngestControlMessage")

    if not issubclass(first_param, IngestControlMessage):
        raise TypeError(f"First parameter must be IngestControlMessage, got {first_param}")

    if not (issubclass(second_param, BaseModel)):
        raise TypeError(f"Second parameter must be a subclass of BaseModel, got {second_param}")

    if not issubclass(return_type, IngestControlMessage):
        raise TypeError(f"Return type must be IngestControlMessage, got {return_type}")


def ingest_callable_signature(sig: inspect.Signature):
    """
    Validates that a callable has the signature:
        (IngestControlMessage) -> IngestControlMessage

    Also allows for generic (*args, **kwargs) signatures for flexibility with class constructors.

    Raises
    ------
    TypeError
        If the signature does not match the expected pattern.
    """
    params = list(sig.parameters.values())

    # If the signature accepts arbitrary keyword arguments, it's flexible enough.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return

    if len(params) != 1:
        raise TypeError(f"Expected exactly 1 parameter, got {len(params)}")

    if params[0].name != "control_message":
        raise TypeError("Expected parameter name: 'control_message'")

    first_param = params[0].annotation
    return_type = sig.return_annotation

    if first_param is inspect.Parameter.empty:
        raise TypeError("Parameter must be annotated with IngestControlMessage")

    if return_type is inspect.Signature.empty:
        raise TypeError("Return type must be annotated with IngestControlMessage")

    # Handle string annotations (forward references)
    if isinstance(first_param, str):
        if first_param != "IngestControlMessage":
            raise TypeError(f"Parameter must be IngestControlMessage, got {first_param}")
    else:
        # Handle actual class annotations
        if not issubclass(first_param, IngestControlMessage):
            raise TypeError(f"Parameter must be IngestControlMessage, got {first_param}")

    # Handle string annotations for return type
    if isinstance(return_type, str):
        if return_type != "IngestControlMessage":
            raise TypeError(f"Return type must be IngestControlMessage, got {return_type}")
    else:
        # Handle actual class annotations
        if not issubclass(return_type, IngestControlMessage):
            raise TypeError(f"Return type must be IngestControlMessage, got {return_type}")
