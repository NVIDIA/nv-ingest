# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema


def valid_module_config():
    """Returns a valid job payload for testing purposes."""
    return {
        "raise_on_failure": True,
    }


def test_task_type_str_bool():
    img_dedup_module_config = valid_module_config()
    img_dedup_module_config["raise_on_failure"] = bool(img_dedup_module_config["raise_on_failure"])
    _ = ImageDedupSchema(**img_dedup_module_config)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_task_type_str_bool_sensitivity(dtype):
    img_dedup_module_config = valid_module_config()
    img_dedup_module_config["raise_on_failure"] = dtype(img_dedup_module_config["raise_on_failure"])

    with pytest.raises(ValidationError):
        _ = ImageDedupSchema(**img_dedup_module_config)
