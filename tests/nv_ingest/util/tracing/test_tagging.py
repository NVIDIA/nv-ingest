# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pytest

from nv_ingest.util.tracing.tagging import traceable
from nv_ingest.util.tracing.tagging import traceable_func


class MockControlMessage:
    def __init__(self):
        self.metadata = {}
        self.timestamp = {}

    def has_metadata(self, key):
        return key in self.metadata

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def set_metadata(self, key, value):
        self.metadata[key] = value

    def get_timestamp(self, key, default=None):
        return self.timestamp.get(key, default)

    def set_timestamp(self, key, value):
        self.timestamp[key] = value

    def filter_timestamp(self, pattern):
        return {k: v for k, v in self.timestamp.items() if pattern in k}


@pytest.fixture
def mock_control_message():
    return MockControlMessage()


# Test with trace tagging enabled and custom trace name
def test_traceable_with_trace_tagging_enabled_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable(trace_name="CustomTrace")
    def sample_function(message):
        pass  # Function body is not relevant for the test

    sample_function(mock_control_message)

    assert mock_control_message.filter_timestamp("trace::entry::CustomTrace")
    assert mock_control_message.filter_timestamp("trace::exit::CustomTrace")
    assert isinstance(mock_control_message.timestamp["trace::entry::CustomTrace"], datetime)
    assert isinstance(mock_control_message.timestamp["trace::exit::CustomTrace"], datetime)


# Test with trace tagging enabled and no custom trace name
def test_traceable_with_trace_tagging_enabled_no_custom_name(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", True)

    @traceable()
    def another_function(message):
        pass  # Function body is not relevant for the test

    another_function(mock_control_message)

    assert mock_control_message.filter_timestamp("trace::entry::another_function")
    assert mock_control_message.filter_timestamp("trace::exit::another_function")
    assert isinstance(mock_control_message.timestamp["trace::entry::another_function"], datetime)
    assert isinstance(mock_control_message.timestamp["trace::exit::another_function"], datetime)


# Test with trace tagging disabled
def test_traceable_with_trace_tagging_disabled(mock_control_message):
    mock_control_message.set_metadata("config::add_trace_tagging", False)

    @traceable()
    def disabled_function(message):
        pass  # Function body is not relevant for the test

    disabled_function(mock_control_message)

    # Ensure no trace metadata was added since trace tagging was disabled
    assert not mock_control_message.filter_timestamp("trace::")


@traceable_func(trace_name="simple_func::{param}")
def simple_func(param, **kwargs):
    return f"Processed {param}"


def test_traceable_func_without_trace_name():
    """
    Test that the traceable_func logs entry and exit times using the function name when no trace_name is provided.
    """
    trace_info = {}
    result = simple_func("sample_value", trace_info=trace_info)

    assert result == "Processed sample_value"
    assert "trace::entry::simple_func::sample_value_0" in trace_info
    assert "trace::exit::simple_func::sample_value_0" in trace_info
    assert isinstance(trace_info["trace::entry::simple_func::sample_value_0"], datetime)
    assert isinstance(trace_info["trace::exit::simple_func::sample_value_0"], datetime)


def test_traceable_func_with_trace_name_formatting():
    """
    Test that the traceable_func logs entry and exit times using the formatted trace_name with argument values.
    """
    trace_info = {}
    result = simple_func("formatted_value", trace_info=trace_info)

    assert result == "Processed formatted_value"
    assert "trace::entry::simple_func::formatted_value_0" in trace_info
    assert "trace::exit::simple_func::formatted_value_0" in trace_info
    assert isinstance(trace_info["trace::entry::simple_func::formatted_value_0"], datetime)
    assert isinstance(trace_info["trace::exit::simple_func::formatted_value_0"], datetime)


def test_traceable_func_dedupe():
    """
    Test that the traceable_func deduplicates trace keys by appending an index when dedupe=True.
    """
    trace_info = {}
    result1 = simple_func("dedupe_test", trace_info=trace_info)
    result2 = simple_func("dedupe_test", trace_info=trace_info)

    assert result1 == "Processed dedupe_test"
    assert result2 == "Processed dedupe_test"

    assert "trace::entry::simple_func::dedupe_test_0" in trace_info
    assert "trace::exit::simple_func::dedupe_test_0" in trace_info
    assert "trace::entry::simple_func::dedupe_test_1" in trace_info
    assert "trace::exit::simple_func::dedupe_test_1" in trace_info

    assert isinstance(trace_info["trace::entry::simple_func::dedupe_test_0"], datetime)
    assert isinstance(trace_info["trace::exit::simple_func::dedupe_test_0"], datetime)
    assert isinstance(trace_info["trace::entry::simple_func::dedupe_test_1"], datetime)
    assert isinstance(trace_info["trace::exit::simple_func::dedupe_test_1"], datetime)


def test_traceable_func_without_trace_info():
    """
    Test that traceable_func creates a new trace_info dictionary if one is not passed.
    """
    result = simple_func("no_trace_info")

    assert result == "Processed no_trace_info"


def test_traceable_func_with_multiple_args():
    """
    Test that traceable_func handles functions with multiple arguments and formats trace_name accordingly.
    """

    @traceable_func(trace_name="multi_args_func::{arg1}::{arg2}")
    def multi_args_func(arg1, arg2, **kwargs):
        return f"Processed {arg1} and {arg2}"

    trace_info = {}
    result = multi_args_func("first_value", "second_value", trace_info=trace_info)

    assert result == "Processed first_value and second_value"
    assert "trace::entry::multi_args_func::first_value::second_value_0" in trace_info
    assert "trace::exit::multi_args_func::first_value::second_value_0" in trace_info


def test_traceable_func_dedupe_disabled():
    """
    Test that the traceable_func does not deduplicate trace keys when dedupe=False.
    """

    @traceable_func(trace_name="no_dedupe_test", dedupe=False)
    def no_dedupe_test(param, **kwargs):
        return f"Processed {param}"

    trace_info = {}
    result1 = no_dedupe_test("no_dedupe", trace_info=trace_info)
    result2 = no_dedupe_test("no_dedupe", trace_info=trace_info)

    assert result1 == "Processed no_dedupe"
    assert result2 == "Processed no_dedupe"

    assert "trace::entry::no_dedupe_test" in trace_info
    assert "trace::exit::no_dedupe_test" in trace_info

    assert "trace::entry::no_dedupe_test_1" not in trace_info
    assert "trace::exit::no_dedupe_test_1" not in trace_info
