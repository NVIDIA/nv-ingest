# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect

import pytest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from nv_ingest_api.util.imports.resolve_callable import resolve_callable_from_path


# Fixture: dynamically create a temporary test module
@pytest.fixture(scope="module")
def test_module():
    with TemporaryDirectory() as tmpdir:
        module_path = Path(tmpdir) / "my_test_module.py"
        module_path.write_text(
            """
def valid_func(x, y): return x + y
not_callable = 42
def missing_param(a): return a
def annotated_func(a: int, b: str) -> int: return a
"""
        )
        sys.path.insert(0, tmpdir)
        yield "my_test_module"
        sys.path.pop(0)
        if "my_test_module" in sys.modules:
            del sys.modules["my_test_module"]


# -- Tests --


def test_valid_func_resolves(test_module):
    fn = resolve_callable_from_path(f"{test_module}:valid_func")
    assert callable(fn)
    assert fn(2, 3) == 5


def test_invalid_path_format():
    with pytest.raises(ValueError, match="expected format 'module.sub:callable'"):
        resolve_callable_from_path("invalidformat")


def test_module_not_found():
    with pytest.raises(ImportError, match="Could not import module 'nonexistent_module'"):
        resolve_callable_from_path("nonexistent_module:some_func")


def test_attribute_not_found(test_module):
    with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
        resolve_callable_from_path(f"{test_module}:does_not_exist")


def test_not_callable(test_module):
    with pytest.raises(TypeError, match="is not callable"):
        resolve_callable_from_path(f"{test_module}:not_callable")


def test_missing_required_params(test_module):
    with pytest.raises(TypeError, match="missing required parameters:"):
        resolve_callable_from_path(f"{test_module}:missing_param", signature_schema=["a", "b"])


def test_signature_schema_callable_return_none_passes(test_module):
    def schema(sig):
        # Implicitly returns None => fail
        return None

    with pytest.raises(TypeError):
        _ = resolve_callable_from_path(f"{test_module}:valid_func", signature_schema=schema)


def test_signature_schema_callable_return_false_fails(test_module):
    def schema(sig):
        # Explicitly return False => fail
        return False

    with pytest.raises(TypeError, match="failed custom signature validation"):
        resolve_callable_from_path(f"{test_module}:valid_func", signature_schema=schema)


def test_signature_schema_callable_raises(test_module):
    def schema(sig: inspect.Signature):
        raise TypeError("Custom schema rejection")

    with pytest.raises(TypeError, match="Custom schema rejection"):
        resolve_callable_from_path(f"{test_module}:valid_func", signature_schema=schema)


def test_signature_schema_validate_annotations(test_module):
    def schema(sig):
        # Require both parameters annotated and return type annotated
        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                return False
        if sig.return_annotation is inspect.Signature.empty:
            return False
        return True

    # annotated_func has annotations => should pass
    fn = resolve_callable_from_path(f"{test_module}:annotated_func", signature_schema=schema)
    assert callable(fn)

    # valid_func has no annotations => should fail
    with pytest.raises(TypeError, match="failed custom signature validation"):
        resolve_callable_from_path(f"{test_module}:valid_func", signature_schema=schema)


def test_invalid_signature_schema_type(test_module):
    with pytest.raises(TypeError, match="expected list of parameter names or callable"):
        resolve_callable_from_path(f"{test_module}:valid_func", signature_schema=123)
