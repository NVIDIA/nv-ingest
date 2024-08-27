# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import TaskInjectionSchema


def test_task_injection_schema_default():
    """
    Test TaskInjectionSchema with default values.
    """
    schema = TaskInjectionSchema()
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_task_injection_schema_explicit_value():
    """
    Test TaskInjectionSchema with an explicit value for raise_on_failure.
    """
    schema = TaskInjectionSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True, "raise_on_failure should respect the explicitly provided value."


def test_task_injection_schema_forbids_extra():
    """
    Test that TaskInjectionSchema forbids extra fields due to the 'extra = "forbid"' configuration.
    """
    with pytest.raises(ValidationError) as excinfo:
        TaskInjectionSchema(raise_on_failure=False, unexpected_field="value")
    assert "extra fields not permitted" in str(excinfo.value), "Schema should not allow extra fields."
