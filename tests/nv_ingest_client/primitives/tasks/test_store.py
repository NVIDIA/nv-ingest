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
        store_method="s3",
        endpoint="minio:9000",
        access_key="foo",
        secret_key="bar",
    )
    assert task._structured
    assert task._images
    assert task._store_method == "s3"
    assert task._extra_params["endpoint"] == "minio:9000"
    assert task._extra_params["access_key"] == "foo"
    assert task._extra_params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreTask(structured=True, images=True, store_method="minio", endpoint="minio:9000")
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
        (None, True, "s3", "foo", "bar"),
        (True, None, "s3", "foo", "bar"),
        (None, None, "minio", "foo", "bar"),
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
        extra_param_1=extra_param_1,
        extra_param_2=extra_param_2,
    )

    expected_dict = {"type": "store", "task_properties": {"params": {}}}

    expected_dict["task_properties"]["structured"] = structured
    expected_dict["task_properties"]["images"] = images
    expected_dict["task_properties"]["method"] = store_method or "minio"
    expected_dict["task_properties"]["params"]["extra_param_1"] = extra_param_1
    expected_dict["task_properties"]["params"]["extra_param_2"] = extra_param_2

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"
