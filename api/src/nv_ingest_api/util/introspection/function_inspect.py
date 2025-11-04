# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for introspecting and analyzing UDF function specifications.
"""

import re
from typing import Optional


def infer_udf_function_name(udf_function: str) -> Optional[str]:
    """
    Attempts to infer the UDF function name from the provided function string.

    Supports three formats:
    1. Inline function: 'def my_func(control_message): ...' -> 'my_func'
    2. Import path: 'my_module.my_function' -> 'my_function'
    3. File path: '/path/to/file.py:function_name' -> 'function_name'

    Parameters
    ----------
    udf_function : str
        The UDF function string.

    Returns
    -------
    Optional[str]
        The inferred UDF function name, or None if inference is not possible.

    Examples
    --------
    >>> infer_udf_function_name("def my_custom_func(control_message): pass")
    'my_custom_func'

    >>> infer_udf_function_name("my_module.submodule.process_data")
    'process_data'

    >>> infer_udf_function_name("/path/to/script.py:custom_function")
    'custom_function'

    >>> infer_udf_function_name("/path/to/script.py")
    None
    """
    udf_function = udf_function.strip()

    # Format 3: File path with explicit function name
    if ":" in udf_function and ("/" in udf_function or "\\" in udf_function):
        # File path with explicit function name: '/path/to/file.py:function_name'
        return udf_function.split(":")[-1].strip()

    # Format 2: Import path like 'module.submodule.function'
    elif "." in udf_function and not udf_function.startswith("def "):
        # Import path: extract the last part as function name
        return udf_function.split(".")[-1].strip()

    # Format 1: Inline function definition
    elif udf_function.startswith("def "):
        # Parse inline function definition to extract function name
        match = re.match(r"def\s+(\w+)\s*\(", udf_function)
        if match:
            return match.group(1)

    return None
