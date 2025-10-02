# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Optional, Type, Union, Callable

from pydantic import BaseModel


def find_pydantic_config_schema(
    actor_class: Type,
    base_class_to_find: Type,
    param_name: str = "config",
) -> Optional[Type[BaseModel]]:
    """
    Introspects a class's MRO to find a Pydantic model in its __init__ signature.

    This function is designed to find the specific Pydantic configuration model
    for a pipeline actor, which might be a direct class or a proxy object.

    Parameters
    ----------
    actor_class : Type
        The actor class or proxy object to inspect.
    base_class_to_find : Type
        The specific base class (e.g., RaySource, RayStage) to look for when
        resolving the true actor class from a proxy.
    param_name : str, optional
        The name of the __init__ parameter to inspect for the Pydantic schema,
        by default "config".

    Returns
    -------
    Optional[Type[BaseModel]]
        The Pydantic BaseModel class if found, otherwise None.
    """
    # 1. Find the actual class to inspect, handling proxy objects.
    cls_to_inspect = None
    if inspect.isclass(actor_class):
        cls_to_inspect = actor_class
    else:
        for base in actor_class.__class__.__mro__:
            if inspect.isclass(base) and issubclass(base, base_class_to_find) and base is not base_class_to_find:
                cls_to_inspect = base
                break

    if not cls_to_inspect:
        return None

    # 2. Walk the MRO of the real class to find the __init__ with the typed parameter.
    for cls in cls_to_inspect.__mro__:
        if param_name in getattr(cls.__init__, "__annotations__", {}):
            try:
                init_sig = inspect.signature(cls.__init__)
                config_param = init_sig.parameters.get(param_name)
                if (
                    config_param
                    and config_param.annotation is not BaseModel
                    and issubclass(config_param.annotation, BaseModel)
                ):
                    return config_param.annotation  # Found the schema
            except (ValueError, TypeError):
                # This class's __init__ is not inspectable (e.g., a C-extension), continue up the MRO.
                continue

    return None


def find_pydantic_config_schema_for_callable(
    callable_fn: Callable,
    param_name: str = "stage_config",
) -> Optional[Type[BaseModel]]:
    """
    Introspects a callable's signature to find a Pydantic model parameter.

    This function is designed to find the specific Pydantic configuration model
    for a pipeline callable function.

    Parameters
    ----------
    callable_fn : Callable
        The callable function to inspect.
    param_name : str, optional
        The name of the parameter to inspect for the Pydantic schema,
        by default "stage_config".

    Returns
    -------
    Optional[Type[BaseModel]]
        The Pydantic BaseModel class if found, otherwise None.
    """
    try:
        sig = inspect.signature(callable_fn)
        config_param = sig.parameters.get(param_name)
        if (
            config_param
            and config_param.annotation is not BaseModel
            and hasattr(config_param.annotation, "__mro__")
            and issubclass(config_param.annotation, BaseModel)
        ):
            return config_param.annotation
    except (ValueError, TypeError):
        # Function signature is not inspectable
        pass

    return None


def find_pydantic_config_schema_unified(
    target: Union[Type, Callable],
    base_class_to_find: Optional[Type] = None,
    param_name: str = "config",
) -> Optional[Type[BaseModel]]:
    """
    Unified function to find Pydantic schema for either classes or callables.

    Parameters
    ----------
    target : Union[Type, Callable]
        The class or callable to inspect.
    base_class_to_find : Optional[Type], optional
        The specific base class to look for when resolving actor classes from proxies.
        Only used for class inspection.
    param_name : str, optional
        The name of the parameter to inspect for the Pydantic schema.
        For classes: defaults to "config"
        For callables: should be "stage_config"

    Returns
    -------
    Optional[Type[BaseModel]]
        The Pydantic BaseModel class if found, otherwise None.
    """
    if callable(target) and not inspect.isclass(target):
        # Handle callable function
        return find_pydantic_config_schema_for_callable(target, param_name)
    elif inspect.isclass(target) or hasattr(target, "__class__"):
        # Handle class or proxy object
        if base_class_to_find is None:
            # If no base class specified, we can't use the original function
            return None
        return find_pydantic_config_schema(target, base_class_to_find, param_name)
    else:
        return None
