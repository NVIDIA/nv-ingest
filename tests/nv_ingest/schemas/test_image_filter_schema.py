# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from pydantic import ValidationError

from nv_ingest.schemas.image_filter_schema import ImageFilterSchema


def valid_module_config():
    """Returns a valid job payload for testing purposes."""
    return {
        "raise_on_failure": True,
        "cpu_only": True,
    }


def test_task_type_str_bool():
    img_filter_module_config = valid_module_config()
    img_filter_module_config["raise_on_failure"] = bool(img_filter_module_config["raise_on_failure"])
    img_filter_module_config["cpu_only"] = bool(img_filter_module_config["cpu_only"])
    _ = ImageFilterSchema(**img_filter_module_config)


@pytest.mark.parametrize("dtype", [int, float, str])
def test_task_type_str_bool_sensitivity(dtype):
    img_filter_module_config = valid_module_config()
    img_filter_module_config["raise_on_failure"] = dtype(img_filter_module_config["raise_on_failure"])
    img_filter_module_config["cpu_only"] = dtype(img_filter_module_config["cpu_only"])

    with pytest.raises(ValidationError):
        _ = ImageFilterSchema(**img_filter_module_config)
