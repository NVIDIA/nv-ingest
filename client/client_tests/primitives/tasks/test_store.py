# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.store import StoreTask

# Initialization and Property Setting


def test_store_task_initialization():
    task = StoreTask(
        structured=True,
        images=True,
        storage_uri="s3://bucket",
        storage_options={"key": "foo"},
        params={
            "access_key": "foo",
            "secret_key": "bar",
            "endpoint": "minio:9000",
        },
    )
    assert task._structured
    assert task._images
    assert task._storage_uri == "s3://bucket"
    assert task._storage_options == {"key": "foo"}
    assert task._params["endpoint"] == "minio:9000"
    assert task._params["access_key"] == "foo"
    assert task._params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreTask(
        structured=True,
        images=True,
        storage_uri="file:///tmp",
        public_base_url="http://public",
        params={"endpoint": "minio:9000"},
    )
    expected_str = (
        "Store Task:\n"
        "  store structured types: True\n"
        "  store image types: True\n"
        "  storage uri: file:///tmp\n"
        "  public base url: http://public\n"
        "  endpoint: minio:9000\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "structured, images, storage_uri, extra_param_1, extra_param_2",
    [
        (True, True, "file:///tmp", "foo", "bar"),
        (False, True, "s3://bucket", "foo", "bar"),
        (False, False, None, "foo", "bar"),
    ],
)
def test_store_task_to_dict(
    structured,
    images,
    storage_uri,
    extra_param_1,
    extra_param_2,
):
    task = StoreTask(
        structured=structured,
        images=images,
        storage_uri=storage_uri,
        storage_options={"region_name": "us-west-2"} if storage_uri else None,
        params={
            "extra_param_1": extra_param_1,
            "extra_param_2": extra_param_2,
        },
    )

    expected_dict = {
        "type": "store",
        "task_properties": {
            "structured": structured,
            "images": images,
            "storage_uri": storage_uri,
            "storage_options": {"region_name": "us-west-2"} if storage_uri else {},
            "public_base_url": None,
            "params": {
                "extra_param_1": extra_param_1,
                "extra_param_2": extra_param_2,
            },
        },
    }

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"


# Default Parameter Handling


def test_store_task_default_params():
    task = StoreTask()
    expected_str_contains = [
        "store structured types: True",
        "store image types: False",
        "storage uri: None",
        "public base url: None",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "store",
        "task_properties": {
            "structured": True,
            "images": False,
            "storage_uri": None,
            "storage_options": {},
            "public_base_url": None,
            "params": {},
        },
    }
    assert task.to_dict() == expected_dict


# Schema Consolidation Tests


def test_store_task_schema_consolidation():
    """Test that StoreTask uses API schema for validation."""
    # Test that valid parameters work
    task = StoreTask(
        structured=False,
        images=True,
        storage_uri="s3://bucket",
        params={"bucket": "test-bucket"},
    )
    assert task._structured is False
    assert task._images is True
    assert task._storage_uri == "s3://bucket"
    assert task._params["bucket"] == "test-bucket"


def test_store_task_extra_params_handling():
    """Test StoreTask handling of extra parameters."""
    task = StoreTask(
        structured=True,
        images=False,
        storage_uri="file:///tmp",
        params={"endpoint": "localhost:9000"},
        access_key="test_key",
        secret_key="test_secret",
    )

    # Extra params should be merged into the main params
    assert task._params["endpoint"] == "localhost:9000"
    assert task._params["access_key"] == "test_key"
    assert task._params["secret_key"] == "test_secret"

    # Test serialization includes all params
    result_dict = task.to_dict()
    assert result_dict["task_properties"]["params"]["endpoint"] == "localhost:9000"
    assert result_dict["task_properties"]["params"]["access_key"] == "test_key"
    assert result_dict["task_properties"]["params"]["secret_key"] == "test_secret"


def test_store_task_edge_cases():
    """Test StoreTask edge cases and boundary conditions."""
    # Test with empty params
    task = StoreTask(
        structured=True,
        images=True,
        params={},
    )
    assert task._params == {}

    # Test serialization of empty params
    result_dict = task.to_dict()
    assert result_dict["task_properties"]["params"] == {}

    # Test with None params (should convert to empty dict)
    task = StoreTask(
        structured=False,
        images=False,
        params=None,
    )
    assert task._params == {}
    assert task.to_dict()["task_properties"]["params"] == {}
