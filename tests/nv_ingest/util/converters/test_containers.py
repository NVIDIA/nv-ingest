# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest.util.converters.containers import merge_dict  # Replace 'your_module' with the actual name of your module


def test_merge_simple_dicts():
    defaults = {"a": 1, "b": 2}
    overrides = {"b": 3, "c": 4}
    expected = {"a": 1, "b": 3, "c": 4}
    assert merge_dict(defaults, overrides) == expected, "Test failed for simple dict merge"


def test_merge_with_nested_dicts():
    defaults = {"a": 1, "b": {"x": 5}}
    overrides = {"b": {"x": 10, "y": 20}, "c": 3}
    expected = {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}
    assert merge_dict(defaults, overrides) == expected, "Test failed for nested dict merge"


def test_merge_with_empty_overrides():
    defaults = {"a": 1, "b": 2}
    overrides = {}
    expected = {"a": 1, "b": 2}
    assert merge_dict(defaults, overrides) == expected, "Test failed when overrides are empty"


def test_merge_with_empty_defaults():
    defaults = {}
    overrides = {"a": 1, "b": 2}
    expected = {"a": 1, "b": 2}
    assert merge_dict(defaults, overrides) == expected, "Test failed when defaults are empty"


def test_merge_overrides_none_values():
    defaults = {"a": 1, "b": 2}
    overrides = {"b": None}
    expected = {"a": 1, "b": None}
    assert merge_dict(defaults, overrides) == expected, "Test failed when overrides contain None values"


def test_merge_does_not_modify_input():
    defaults = {"a": 1, "b": 2}
    overrides = {"b": 3}
    merge_dict(defaults.copy(), overrides.copy())
    assert defaults == {"a": 1, "b": 2} and overrides == {"b": 3}, "Test failed: function should not modify inputs"


def test_merge_with_complex_nested_dicts():
    defaults = {"a": 1, "b": {"x": {"alpha": "beta"}, "y": 99}}
    overrides = {"b": {"x": {"gamma": "delta"}, "z": 100}}
    expected = {"a": 1, "b": {"x": {"alpha": "beta", "gamma": "delta"}, "y": 99, "z": 100}}
    assert merge_dict(defaults, overrides) == expected, "Test failed for complex nested dict merge"


def test_merge_with_lists_should_not_merge():
    defaults = {"a": [1, 2], "b": 2}
    overrides = {"a": [3, 4]}
    expected = {"a": [3, 4], "b": 2}
    assert merge_dict(defaults, overrides) == expected, "Test failed: lists should be replaced, not merged"


def test_merge_with_differing_types():
    defaults = {"a": 1, "b": {"x": 10}}
    overrides = {"b": "new_string"}
    expected = {"a": 1, "b": "new_string"}
    assert merge_dict(defaults, overrides) == expected, "Test failed for merging differing types"
