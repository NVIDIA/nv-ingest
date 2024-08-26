# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import MetadataInjectorSchema


def test_metadata_injector_schema_default():
    """
    Test the MetadataInjectorSchema with default values.
    """
    schema = MetadataInjectorSchema()
    assert schema.raise_on_failure is False, "Default value for raise_on_failure should be False."


def test_metadata_injector_schema_explicit_value():
    """
    Test the MetadataInjectorSchema with an explicit value for raise_on_failure.
    """
    schema = MetadataInjectorSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True, "raise_on_failure should respect the explicitly provided value."


def test_metadata_injector_schema_forbids_extra():
    """
    Test that the MetadataInjectorSchema forbids extra fields due to the 'extra = "forbid"' configuration.
    """
    with pytest.raises(ValidationError) as excinfo:
        MetadataInjectorSchema(raise_on_failure=False, unexpected_field="value")
    assert "extra fields not permitted" in str(excinfo.value), "Schema should not allow extra fields."


@pytest.mark.parametrize("input_value", [True, False])
def test_metadata_injector_schema_raise_on_failure_parametrized(input_value):
    """
    Parametrized test for different boolean values of raise_on_failure.
    """
    schema = MetadataInjectorSchema(raise_on_failure=input_value)
    assert schema.raise_on_failure is input_value, f"raise_on_failure should be {input_value}."
