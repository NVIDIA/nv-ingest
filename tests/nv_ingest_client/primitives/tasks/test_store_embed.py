# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.primitives.tasks.store import StoreEmbedTask

# Initialization and Property Setting


def test_store_task_initialization():
    task = StoreEmbedTask(
        params={
            "endpoint": "minio:9000",
            "access_key": "foo",
            "secret_key": "bar",
        }
    )
    assert task._params["endpoint"] == "minio:9000"
    assert task._params["access_key"] == "foo"
    assert task._params["secret_key"] == "bar"


# String Representation Tests


def test_store_task_str_representation():
    task = StoreEmbedTask(
        params={
            "endpoint": "minio:9000"
        }
    )
    expected_str = (
        "Store Embed Task:\n"
        "  endpoint: minio:9000\n"
    )
    assert str(task) == expected_str


# Dictionary Representation Tests


@pytest.mark.parametrize(
    "extra_param_1, extra_param_2",
    [
        ("foo", "bar"),
    ],
)
def test_store_task_to_dict(
    extra_param_1,
    extra_param_2,
):
    task = StoreEmbedTask(
        params={
            "extra_param_1": extra_param_1,
            "extra_param_2": extra_param_2,
        }
    )

    expected_dict = {"type": "store_embedding", "task_properties": {"params": {}}}

    expected_dict["task_properties"]["params"]["extra_param_1"] = extra_param_1
    expected_dict["task_properties"]["params"]["extra_param_2"] = extra_param_2

    assert task.to_dict() == expected_dict, "The to_dict method did not return the expected dictionary representation"
