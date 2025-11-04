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
        params={
            "access_key": "foo",
            "secret_key": "bar",
            "endpoint": "minio:9000",
        },
        store_method="s3",
    )
    assert task._structured
    assert task._images
    assert task._store_method == "s3"
    assert task._params["endpoint"] == "minio:9000"
    assert task._params["access_key"] == "foo"
    assert task._params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreTask(structured=True, images=True, store_method="minio", params={"endpoint": "minio:9000"})
    expected_str = (
        "Store Task:\n"
        "  store structured types: True\n"
        "  store image types: True\n"
        "  store method: minio\n"
        "  endpoint: minio:9000\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "structured, images, store_method, extra_param_1, extra_param_2",
    [
        (True, True, "minio", "foo", "bar"),
        (False, True, "minio", "foo", "bar"),
        (False, False, "minio", "foo", "bar"),
        (False, True, "s3", "foo", "bar"),
        (True, False, "s3", "foo", "bar"),
    ],
)
def test_store_task_to_dict(
    structured,
    images,
    store_method,
    extra_param_1,
    extra_param_2,
):
    task = StoreTask(
        structured=structured,
        images=images,
        store_method=store_method,
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
            "method": store_method or "minio",
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
        "store method: minio",
    ]
    for expected_part in expected_str_contains:
        assert expected_part in str(task)

    expected_dict = {
        "type": "store",
        "task_properties": {
            "structured": True,
            "images": False,
            "method": "minio",
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
        store_method="s3",
        params={"bucket": "test-bucket"},
    )
    assert task._structured is False
    assert task._images is True
    assert task._store_method == "s3"
    assert task._params["bucket"] == "test-bucket"


def test_store_task_extra_params_handling():
    """Test StoreTask handling of extra parameters."""
    task = StoreTask(
        structured=True,
        images=False,
        store_method="minio",
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
        store_method="minio",
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
        store_method="s3",
        params=None,
    )
    assert task._params == {}
    assert task.to_dict()["task_properties"]["params"] == {}
