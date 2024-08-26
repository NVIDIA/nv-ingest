# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel
from pydantic import Field

from nv_ingest.util.schema.schema_validator import validate_schema


class UserSchema(BaseModel):
    name: str
    age: int = Field(..., gt=0)  # Age must be greater than 0


def test_validate_schema_success():
    """Test validating metadata with a correct structure."""
    user_metadata = {"name": "John Doe", "age": 30}
    validated_user = validate_schema(user_metadata, UserSchema)
    assert validated_user.name == "John Doe"
    assert validated_user.age == 30


def test_validate_schema_failure():
    """Test validating metadata that does not conform to the schema."""
    incorrect_metadata = {"name": "John Doe", "age": -5}  # Invalid age
    with pytest.raises(ValueError):
        validate_schema(incorrect_metadata, UserSchema)


def test_validate_schema_optional_and_default():
    """Test schema validation with optional fields and fields with default values."""
    # Assuming the UserSchema has optional fields or fields with default values
    partial_metadata = {"name": "Jane Doe"}  # 'age' is intentionally omitted
    with pytest.raises(ValueError):  # Expecting failure due to missing required 'age'
        validate_schema(partial_metadata, UserSchema)


def test_validate_schema_with_extra_fields():
    """Test schema validation with extra fields not defined in the schema."""
    extra_metadata = {"name": "John Doe", "age": 30, "title": "sir"}
    # Depending on schema configuration, this might either pass or fail
    # Assuming extra fields are ignored
    validated_user = validate_schema(extra_metadata, UserSchema)
    assert validated_user.name == "John Doe"
    assert validated_user.age == 30
    with pytest.raises(AttributeError):  # Extra fields should not be set
        _ = validated_user.title
