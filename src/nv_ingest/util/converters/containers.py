# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Any
from typing import Dict

logger = logging.getLogger(__name__)


def merge_dict(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries, with values from the `overrides` dictionary taking precedence.

    This function merges the `overrides` dictionary into the `defaults` dictionary. If a key in both dictionaries
    has a dictionary as its value, the function will recursively merge those dictionaries. Otherwise, the value
    from the `overrides` dictionary will overwrite the value in the `defaults` dictionary.

    Parameters
    ----------
    defaults : dict of {str: Any}
        The default dictionary that will be updated with values from the `overrides` dictionary.
    overrides : dict of {str: Any}
        The dictionary containing values that will override or extend those in the `defaults` dictionary.

    Returns
    -------
    dict of {str: Any}
        The merged dictionary, with values from the `overrides` dictionary taking precedence.

    Examples
    --------
    >>> defaults = {
    ...     "a": 1,
    ...     "b": {
    ...         "c": 3,
    ...         "d": 4
    ...     },
    ...     "e": 5
    ... }
    >>> overrides = {
    ...     "b": {
    ...         "c": 30
    ...     },
    ...     "f": 6
    ... }
    >>> result = merge_dict(defaults, overrides)
    >>> result
    {'a': 1, 'b': {'c': 30, 'd': 4}, 'e': 5, 'f': 6}

    Notes
    -----
    - The `merge_dict` function modifies the `defaults` dictionary in place. If you need to preserve the original
      `defaults` dictionary, consider passing a copy instead.
    - This function is particularly useful when combining configuration dictionaries where certain settings should
      override defaults.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            defaults[key] = merge_dict(defaults.get(key, {}), value)
        else:
            defaults[key] = overrides[key]
    return defaults
