# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.store import StoreEmbedTask

# Initialization and Property Setting


def test_store_task_initialization():
    task = StoreEmbedTask(
        embedding=True,
        store_method="s3",
        extra_params={
            "endpoint": "minio:9000",
            "access_key": "foo",
            "secret_key": "bar",
        }
    )
    assert task._embedding
    assert task._store_method == "s3"
    assert task._extra_params["endpoint"] == "minio:9000"
    assert task._extra_params["access_key"] == "foo"
    assert task._extra_params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreEmbedTask(
        embedding=True, 
        store_method="minio",
        extra_params={
            "endpoint": "minio:9000"
        }
    )
    expected_str = (
        "Store Embed Task:\n"
        "  store embedding: True\n"
        "  store method: minio\n"
        "  endpoint: minio:9000\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "embedding, store_method, extra_param_1, extra_param_2",
    [
        (True, "minio", "foo", "bar"),
        (True, "minio", "foo", "bar"),
        (False, "minio", "foo", "bar"),
        (True, "s3", "foo", "bar"),
        (True, "s3", "foo", "bar"),
        (None, "s3", "foo", "bar"),
        (None, "minio", "foo", "bar"),
    ],
)
def test_store_task_to_dict(
    embedding,
    store_method,
    extra_param_1,
    extra_param_2,
):
    task = StoreEmbedTask(
        embedding=embedding,
        store_method=store_method,
        extra_params={
            "extra_param_1": extra_param_1,
            "extra_param_2": extra_param_2,
        }
    )

    expected_dict = {"type": "store_embedding", "task_properties": {"extra_params": {}}}

    expected_dict["task_properties"]["embedding"] = embedding
    expected_dict["task_properties"]["method"] = store_method or "minio"
    expected_dict["task_properties"]["extra_params"]["extra_param_1"] = extra_param_1
    expected_dict["task_properties"]["extra_params"]["extra_param_2"] = extra_param_2

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"
