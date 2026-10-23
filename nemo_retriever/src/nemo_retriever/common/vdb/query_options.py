# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation shared by typed requests and LanceDB search entrypoints."""

from typing import Any


def validate_hnsw_ef(value: Any) -> int | None:
    """Keep an omitted HNSW budget unset; accept only explicit positive integers."""
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("ef must be a positive integer or None.")
    return value
