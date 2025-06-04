# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
from typing import Callable, Optional, Union, List


def resolve_callable_from_path(
    path: str, signature_schema: Optional[Union[List[str], Callable[[inspect.Signature], bool]]] = None
) -> Callable:
    """
    Import and return a callable from a module path string like 'module.submodule:callable_name'.

    Parameters
    ----------
    path : str
        The module path and callable in the format 'module.sub:callable'.
    signature_schema : Optional[Union[List[str], Callable]]
        If a list of parameter names is given, ensures the callable has parameters with those names.
        If a callable is provided, it should take an `inspect.Signature` and return True/False.

    Returns
    -------
    Callable
        The resolved and validated callable.

    Raises
    ------
    ValueError
        If the path is not correctly formatted.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the attribute does not exist in the module.
    TypeError
        If the resolved attribute is not callable or the signature does not match.
    """
    if ":" not in path:
        raise ValueError(f"Invalid path '{path}': expected format 'module.sub:callable'")

    module_path, attr_name = path.split(":", 1)

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import module '{module_path}'") from e

    try:
        obj = getattr(mod, attr_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'") from e

    if not callable(obj):
        raise TypeError(f"Object '{attr_name}' from module '{module_path}' is not callable")

    if signature_schema:
        sig = inspect.signature(obj)

        if isinstance(signature_schema, list):
            actual_params = list(sig.parameters.keys())
            missing = [p for p in signature_schema if p not in actual_params]
            if missing:
                raise TypeError(f"Callable '{attr_name}' is missing required parameters: {missing}")
        elif callable(signature_schema):
            if not signature_schema(sig):
                raise TypeError(f"Callable '{attr_name}' failed custom signature validation: {sig}")
        else:
            raise TypeError(
                f"Invalid signature_schema: expected list of parameter names or callable, got {type(signature_schema)}"
            )

    return obj
