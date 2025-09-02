# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
from typing import Callable, Union, List, Optional

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage


def resolve_obj_from_path(path: str, allowed_base_paths: Optional[List[str]] = None) -> object:
    """
    Import and return an object from a string path of the form 'module.sub:attr'.

    To enhance security, this function can restrict imports to a list of allowed base module paths.
    """
    if ":" not in path:
        raise ValueError(f"Invalid path '{path}': expected format 'module.sub:attr'")
    module_path, attr_name = path.split(":", 1)

    # Security check: only allow imports from specified base paths if provided.
    if allowed_base_paths:
        is_allowed = any(module_path == base or module_path.startswith(base + ".") for base in allowed_base_paths)
        if not is_allowed:
            raise ImportError(
                f"Module '{module_path}' is not in the list of allowed base paths. "
                f"Allowed paths: {allowed_base_paths}"
            )

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module '{module_path}'") from e
    try:
        obj = getattr(mod, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'") from e
    return obj


def resolve_callable_from_path(
    callable_path: str,
    signature_schema: Union[List[str], Callable[[inspect.Signature], None], str],
    allowed_base_paths: Optional[List[str]] = None,
) -> Callable:
    """
    Import and return a callable from a module path string like 'module.submodule:callable_name',
    and validate its signature using the required signature_schema (callable or path to callable).

    Parameters
    ----------
    callable_path : str
        The module path and callable in the format 'module.sub:callable'.
    signature_schema : Union[List[str], Callable, str]
        Either:
            - A list of parameter names to require.
            - A callable that takes an inspect.Signature and raises on failure.
            - A string path to such a callable ('module.sub:schema_checker').
    allowed_base_paths : Optional[List[str]]
        An optional list of base module paths from which imports are allowed.
        If provided, both the callable and any signature schema specified by path
        must reside within one of these paths.

    Returns
    -------
    Callable
        The resolved and validated callable.

    Raises
    ------
    ValueError
        If the path is not correctly formatted.
    ImportError
        If the module cannot be imported or is not in the allowed paths.
    AttributeError
        If the attribute does not exist in the module.
    TypeError
        If the resolved attribute is not callable or the signature does not match.
    """
    obj = resolve_obj_from_path(callable_path, allowed_base_paths=allowed_base_paths)
    if not callable(obj):
        raise TypeError(f"Object '{callable_path}' is not callable")

    # Load/check signature_schema
    schema_checker = signature_schema
    if isinstance(signature_schema, str):
        # When loading the schema checker, apply the same security restrictions.
        schema_checker = resolve_obj_from_path(signature_schema, allowed_base_paths=allowed_base_paths)

    sig = inspect.signature(obj)
    if isinstance(schema_checker, list):
        actual_params = list(sig.parameters.keys())
        missing = [p for p in schema_checker if p not in actual_params]
        if missing:
            raise TypeError(
                f"Callable at '{callable_path}' is missing required parameters: {missing}\n"
                f"Actual parameters: {actual_params}"
            )
    elif callable(schema_checker):
        try:
            schema_checker(sig)
        except Exception as e:
            raise TypeError(f"Signature validation for '{callable_path}' failed: {e}") from e
    else:
        raise TypeError(f"Invalid signature_schema: expected list, callable, or str, got {type(signature_schema)}")

    return obj


def resolve_actor_class_from_path(
    path: str, expected_base_class: type, allowed_base_paths: Optional[List[str]] = None
) -> type:
    """
    Resolves an actor class from a path and validates that it is a class
    that inherits from the expected base class. This function correctly handles
    decorated Ray actors by inspecting their original class.

    Parameters
    ----------
    path : str
        The full import path to the actor class.
    expected_base_class : type
        The base class that the resolved class must inherit from.
    allowed_base_paths : Optional[List[str]]
        An optional list of base module paths from which imports are allowed.

    Returns
    -------
    type
        The resolved actor class (or Ray actor factory).
    """
    obj = resolve_obj_from_path(path, allowed_base_paths=allowed_base_paths)

    # Determine the class to validate. If it's a Ray actor factory, we need to
    # inspect its MRO to find the original user-defined class.
    cls_to_validate = None
    if inspect.isclass(obj):
        cls_to_validate = obj
    else:
        # For actor factories, find the base class in the MRO that inherits from RayActorStage
        for base in obj.__class__.__mro__:
            if inspect.isclass(base) and issubclass(base, RayActorStage) and base is not RayActorStage:
                cls_to_validate = base
                break

    if cls_to_validate is None:
        raise TypeError(
            f"Could not resolve a valid actor class from path '{path}'. "
            f"The object is not a class and not a recognized actor factory."
        )

    if not issubclass(cls_to_validate, expected_base_class):
        raise TypeError(
            f"Actor class '{cls_to_validate.__name__}' at '{path}' must inherit from '{expected_base_class.__name__}'."
        )

    return obj
