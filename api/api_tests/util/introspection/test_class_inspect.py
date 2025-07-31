# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import pytest
from typing import Optional
from unittest.mock import Mock

from pydantic import BaseModel, Field

from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.introspection.class_inspect import (
    find_pydantic_config_schema,
    find_pydantic_config_schema_for_callable,
    find_pydantic_config_schema_unified,
)


# Test fixtures - Mock classes and schemas for testing
class MockBaseClass:
    """Mock base class for testing."""

    pass


class MockSchema(BaseModel):
    """Mock Pydantic schema for testing."""

    test_field: str = Field("default", description="Test field")


class MockSchemaWithValidation(BaseModel):
    """Mock Pydantic schema with validation for testing."""

    required_field: str = Field(..., description="Required field")
    optional_field: Optional[int] = Field(None, description="Optional field")


class MockActorWithSchema:
    """Mock actor class with Pydantic schema in __init__."""

    def __init__(self, config: MockSchema):
        self.config = config


class MockActorWithValidationSchema:
    """Mock actor class with validation schema in __init__."""

    def __init__(self, config: MockSchemaWithValidation):
        self.config = config


class MockActorWithoutSchema:
    """Mock actor class without Pydantic schema in __init__."""

    def __init__(self, config: dict):
        self.config = config


class MockActorWithBaseModelSchema:
    """Mock actor class with BaseModel (not subclass) in __init__."""

    def __init__(self, config: BaseModel):
        self.config = config


class MockInheritedActor(MockBaseClass):
    """Mock actor that inherits from base class."""

    def __init__(self, config: MockSchema):
        self.config = config


# Test callables
def mock_callable_with_schema(control_message, stage_config: MockSchema):
    """Mock callable with Pydantic schema."""
    return control_message


def mock_callable_with_validation_schema(control_message, stage_config: MockSchemaWithValidation):
    """Mock callable with validation schema."""
    return control_message


def mock_callable_without_schema(control_message, stage_config: dict):
    """Mock callable without Pydantic schema."""
    return control_message


def mock_callable_with_base_model(control_message, stage_config: BaseModel):
    """Mock callable with BaseModel (not subclass)."""
    return control_message


def mock_callable_no_stage_config(control_message):
    """Mock callable without stage_config parameter."""
    return control_message


def mock_callable_wrong_param_name(control_message, config: MockSchema):
    """Mock callable with wrong parameter name."""
    return control_message


class TestFindPydanticConfigSchema:
    """Test cases for find_pydantic_config_schema function."""

    def test_find_schema_in_direct_class(self):
        """Test finding schema in a direct class."""
        schema = find_pydantic_config_schema(MockActorWithSchema, MockBaseClass)
        assert schema == MockSchema

    def test_find_schema_in_inherited_class(self):
        """Test finding schema in an inherited class."""
        schema = find_pydantic_config_schema(MockInheritedActor, MockBaseClass)
        assert schema == MockSchema

    def test_find_schema_with_validation(self):
        """Test finding schema with validation fields."""
        schema = find_pydantic_config_schema(MockActorWithValidationSchema, MockBaseClass)
        assert schema == MockSchemaWithValidation

    def test_no_schema_found_dict_config(self):
        """Test when no Pydantic schema is found (dict config)."""
        schema = find_pydantic_config_schema(MockActorWithoutSchema, MockBaseClass)
        assert schema is None

    def test_no_schema_found_base_model(self):
        """Test when BaseModel (not subclass) is used."""
        schema = find_pydantic_config_schema(MockActorWithBaseModelSchema, MockBaseClass)
        assert schema is None

    def test_custom_param_name(self):
        """Test finding schema with custom parameter name."""

        class CustomParamActor:
            def __init__(self, custom_config: MockSchema):
                self.custom_config = custom_config

        schema = find_pydantic_config_schema(CustomParamActor, MockBaseClass, param_name="custom_config")
        assert schema == MockSchema

    def test_non_class_input(self):
        """Test with non-class input."""
        schema = find_pydantic_config_schema("not_a_class", MockBaseClass)
        assert schema is None

    def test_uninspectable_init(self):
        """Test with class that has uninspectable __init__."""

        class UninspectableActor:
            # Mock uninspectable __init__ by removing annotations
            def __init__(self, config):
                pass

        # Remove annotations to simulate C-extension
        if hasattr(UninspectableActor.__init__, "__annotations__"):
            delattr(UninspectableActor.__init__, "__annotations__")

        schema = find_pydantic_config_schema(UninspectableActor, MockBaseClass)
        assert schema is None


class TestFindPydanticConfigSchemaForCallable:
    """Test cases for find_pydantic_config_schema_for_callable function."""

    def test_find_schema_in_callable(self):
        """Test finding schema in a callable function."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_with_schema)
        assert schema == MockSchema

    def test_find_schema_with_validation(self):
        """Test finding schema with validation in callable."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_with_validation_schema)
        assert schema == MockSchemaWithValidation

    def test_no_schema_found_dict_config(self):
        """Test when no Pydantic schema is found in callable (dict config)."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_without_schema)
        assert schema is None

    def test_no_schema_found_base_model(self):
        """Test when BaseModel (not subclass) is used in callable."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_with_base_model)
        assert schema is None

    def test_custom_param_name(self):
        """Test finding schema with custom parameter name."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_wrong_param_name, param_name="config")
        assert schema == MockSchema

    def test_missing_param(self):
        """Test when the expected parameter is missing."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_no_stage_config)
        assert schema is None

    def test_wrong_param_name(self):
        """Test when parameter name doesn't match expected."""
        schema = find_pydantic_config_schema_for_callable(mock_callable_wrong_param_name)
        assert schema is None

    def test_uninspectable_callable(self):
        """Test with uninspectable callable."""
        # Create a mock callable that raises ValueError on signature inspection
        mock_callable = Mock()
        mock_callable.__name__ = "mock_callable"

        # Mock inspect.signature to raise ValueError
        original_signature = inspect.signature

        def mock_signature(obj):
            if obj is mock_callable:
                raise ValueError("Cannot inspect signature")
            return original_signature(obj)

        inspect.signature = mock_signature
        try:
            schema = find_pydantic_config_schema_for_callable(mock_callable)
            assert schema is None
        finally:
            inspect.signature = original_signature


class TestFindPydanticConfigSchemaUnified:
    """Test cases for find_pydantic_config_schema_unified function."""

    def test_unified_with_class(self):
        """Test unified function with class input."""
        schema = find_pydantic_config_schema_unified(MockActorWithSchema, MockBaseClass, param_name="config")
        assert schema == MockSchema

    def test_unified_with_callable(self):
        """Test unified function with callable input."""
        schema = find_pydantic_config_schema_unified(mock_callable_with_schema, param_name="stage_config")
        assert schema == MockSchema

    def test_unified_callable_without_base_class(self):
        """Test unified function with callable and no base class."""
        schema = find_pydantic_config_schema_unified(
            mock_callable_with_schema, base_class_to_find=None, param_name="stage_config"
        )
        assert schema == MockSchema

    def test_unified_class_without_base_class(self):
        """Test unified function with class but no base class."""
        schema = find_pydantic_config_schema_unified(MockActorWithSchema, base_class_to_find=None, param_name="config")
        assert schema is None

    def test_unified_with_invalid_input(self):
        """Test unified function with invalid input."""
        schema = find_pydantic_config_schema_unified("not_class_or_callable")
        assert schema is None

    def test_unified_callable_detection(self):
        """Test that unified function correctly detects callables vs classes."""
        # Test with callable
        callable_schema = find_pydantic_config_schema_unified(mock_callable_with_schema, param_name="stage_config")
        assert callable_schema == MockSchema

        # Test with class
        class_schema = find_pydantic_config_schema_unified(MockActorWithSchema, MockBaseClass, param_name="config")
        assert class_schema == MockSchema

    def test_unified_class_that_is_also_callable(self):
        """Test unified function with class that is also callable."""

        class CallableClass:
            def __init__(self, config: MockSchema):
                self.config = config

            def __call__(self, control_message, stage_config: MockSchemaWithValidation):
                return control_message

        # Should be treated as a class since inspect.isclass returns True
        schema = find_pydantic_config_schema_unified(CallableClass, MockBaseClass, param_name="config")
        assert schema == MockSchema


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_schema_with_no_mro(self):
        """Test with schema that doesn't inherit from BaseModel."""

        class NonBaseModelSchema:
            """A schema that doesn't inherit from BaseModel."""

            pass

        def test_callable(
            control_message: IngestControlMessage, stage_config: NonBaseModelSchema
        ) -> IngestControlMessage:
            return control_message

        schema = find_pydantic_config_schema_for_callable(test_callable)
        assert schema is None

    def test_schema_inheritance_chain(self):
        """Test schema detection through inheritance chain."""

        class BaseActor:
            def __init__(self, config: dict):
                pass

        class MiddleActor(BaseActor):
            pass

        class FinalActor(MiddleActor):
            def __init__(self, config: MockSchema):
                super().__init__(config)

        schema = find_pydantic_config_schema(FinalActor, MockBaseClass)
        assert schema == MockSchema

    def test_multiple_config_params(self):
        """Test class with multiple config-like parameters."""

        class MultiConfigActor:
            def __init__(self, config: MockSchema, other_config: dict):
                self.config = config
                self.other_config = other_config

        schema = find_pydantic_config_schema(MultiConfigActor, MockBaseClass)
        assert schema == MockSchema

    def test_schema_with_generic_types(self):
        """Test schema detection with generic types."""
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class GenericSchema(BaseModel, Generic[T]):
            data: T

        class GenericActor:
            def __init__(self, config: GenericSchema[str]):
                self.config = config

        # This should handle generic types gracefully
        _ = find_pydantic_config_schema(GenericActor, MockBaseClass)
        # Note: Generic types might not be detected as BaseModel subclasses
        # This test ensures no exceptions are raised


if __name__ == "__main__":
    pytest.main([__file__])
