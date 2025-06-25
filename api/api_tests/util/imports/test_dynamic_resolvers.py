# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import types
import pytest
import sys

from nv_ingest_api.util.imports.dynamic_resolvers import resolve_obj_from_path, resolve_callable_from_path


@pytest.fixture
def dummy_module(monkeypatch):
    mod = types.ModuleType("dummy_mod")

    def foo(a, b):
        return a + b

    def bar():
        return "bar"

    def sig_checker_good(sig):
        params = list(sig.parameters.keys())
        if params != ["a", "b"]:
            raise TypeError("Expected ['a', 'b']")

    def sig_checker_bad(sig):
        raise TypeError("Always fails!")

    mod.foo = foo
    mod.bar = bar
    mod.not_callable = 42
    mod.sig_checker_good = sig_checker_good
    mod.sig_checker_bad = sig_checker_bad
    sys.modules["dummy_mod"] = mod
    yield mod
    del sys.modules["dummy_mod"]


# -------- resolve_obj_from_path --------


def test_resolve_obj_from_path_success(dummy_module):
    obj = resolve_obj_from_path("dummy_mod:foo")
    assert callable(obj)
    assert obj is dummy_module.foo


def test_resolve_obj_from_path_missing_colon():
    with pytest.raises(ValueError):
        resolve_obj_from_path("dummy_mod.foo")


def test_resolve_obj_from_path_import_error():
    with pytest.raises(ImportError):
        resolve_obj_from_path("not_a_module:foo")


def test_resolve_obj_from_path_attribute_error(dummy_module):
    with pytest.raises(AttributeError):
        resolve_obj_from_path("dummy_mod:not_a_real_attr")


# -------- resolve_callable_from_path: param list --------


def test_resolve_callable_from_path_with_param_list(dummy_module):
    # foo(a, b) matches ["a", "b"]
    fn = resolve_callable_from_path("dummy_mod:foo", ["a", "b"])
    assert callable(fn)
    assert fn(2, 3) == 5


def test_resolve_callable_from_path_param_list_missing(dummy_module):
    # foo(a, b) does not have 'c'
    with pytest.raises(TypeError) as exc:
        resolve_callable_from_path("dummy_mod:foo", ["a", "b", "c"])
    assert "missing required parameters" in str(exc.value)


# -------- resolve_callable_from_path: callable signature checker --------


def test_resolve_callable_from_path_callable_checker_good(dummy_module):
    fn = resolve_callable_from_path("dummy_mod:foo", dummy_module.sig_checker_good)
    assert callable(fn)
    assert fn(3, 4) == 7


def test_resolve_callable_from_path_callable_checker_bad(dummy_module):
    with pytest.raises(TypeError) as exc:
        resolve_callable_from_path("dummy_mod:foo", dummy_module.sig_checker_bad)
    assert "failed custom signature validation" in str(exc.value)


# -------- resolve_callable_from_path: checker as string --------


def test_resolve_callable_from_path_checker_as_string_good(dummy_module):
    fn = resolve_callable_from_path("dummy_mod:foo", "dummy_mod:sig_checker_good")
    assert callable(fn)


def test_resolve_callable_from_path_checker_as_string_bad(dummy_module):
    with pytest.raises(TypeError):
        resolve_callable_from_path("dummy_mod:foo", "dummy_mod:sig_checker_bad")


def test_resolve_callable_from_path_invalid_signature_schema_type(dummy_module):
    with pytest.raises(TypeError):
        resolve_callable_from_path("dummy_mod:foo", 12345)


def test_resolve_callable_from_path_not_callable(dummy_module):
    with pytest.raises(TypeError):
        resolve_callable_from_path("dummy_mod:not_callable", ["x"])


def test_resolve_callable_from_path_missing_colon_in_signature_schema(dummy_module):
    # Should raise ValueError for missing colon in schema checker path
    with pytest.raises(ValueError):
        resolve_callable_from_path("dummy_mod:foo", "dummy_mod.sig_checker_good")


def test_resolve_obj_from_path_math_sqrt():
    obj = resolve_obj_from_path("math:sqrt")
    assert callable(obj)
    assert obj(9) == 3.0


def test_resolve_obj_from_path_operator_add():
    obj = resolve_obj_from_path("operator:add")
    assert callable(obj)
    assert obj(2, 3) == 5


def test_resolve_callable_from_path_with_signature_schema_list_math_pow():
    # math.pow(x, y): check for params x and y
    fn = resolve_callable_from_path("math:pow", ["x", "y"])
    assert callable(fn)
    assert fn(2, 3) == 8.0


def test_resolve_callable_from_path_with_signature_schema_list_operator_getitem():
    # operator.getitem(a, b): check for params a and b
    fn = resolve_callable_from_path("operator:getitem", ["a", "b"])
    assert callable(fn)
    assert fn([10, 20, 30], 2) == 30


def test_resolve_callable_from_path_with_callable_schema_inspect_signature():
    # inspect.signature(obj) should have param 'obj'
    def has_obj_param(sig):
        params = list(sig.parameters.keys())
        print(params)
        if params[0] != "obj":
            raise TypeError("Expected first parameter 'obj'")

    fn = resolve_callable_from_path("inspect:signature", has_obj_param)
    assert callable(fn)
    sig = fn(len)
    assert isinstance(sig, inspect.Signature)


def test_resolve_callable_from_path_builtin_type_error():
    # math.sqrt only accepts one arg, so this will fail for a param list ['x', 'y']
    with pytest.raises(TypeError):
        resolve_callable_from_path("math:sqrt", ["x", "y"])


def test_resolve_obj_from_path_error_on_builtin_missing():
    with pytest.raises(AttributeError):
        resolve_obj_from_path("math:does_not_exist")


def test_resolve_obj_from_path_error_on_invalid_format():
    with pytest.raises(ValueError):
        resolve_obj_from_path("mathsqrt")


def test_resolve_callable_from_path_with_str_schema_checker_operator_add():
    # We use a checker that expects params ['a', 'b'], as a string path
    mod = types.ModuleType("checker_mod")

    def checker(sig):
        if list(sig.parameters.keys()) != ["a", "b"]:
            raise TypeError("Params not a, b")

    mod.checker = checker
    import sys

    sys.modules["checker_mod"] = mod
    try:
        fn = resolve_callable_from_path("operator:add", "checker_mod:checker")
        assert fn(5, 6) == 11
    finally:
        del sys.modules["checker_mod"]


def test_resolve_callable_from_path_with_bad_schema_type():
    with pytest.raises(TypeError):
        resolve_callable_from_path("os:getcwd", 123)


# Tests for allowed_base_paths security feature


def test_resolve_obj_from_path_allowed():
    """Test that an object is resolved when its path is in the allowed list."""
    obj = resolve_obj_from_path("os:getcwd", allowed_base_paths=["os"])
    import os

    assert obj == os.getcwd


def test_resolve_obj_from_path_allowed_submodule():
    """Test that an object in a submodule is resolved when the base path is allowed."""
    obj = resolve_obj_from_path("os.path:join", allowed_base_paths=["os"])
    import os.path

    assert obj == os.path.join


def test_resolve_obj_from_path_denied():
    """Test that an ImportError is raised when the path is not in the allowed list."""
    with pytest.raises(ImportError, match="is not in the list of allowed base paths"):
        resolve_obj_from_path("sys:executable", allowed_base_paths=["os"])


def test_resolve_obj_from_path_no_restrictions():
    """Test that the resolver works without restrictions if no allowed paths are given."""
    obj = resolve_obj_from_path("os:getcwd")
    import os

    assert obj == os.getcwd


def test_resolve_callable_from_path_allowed():
    """Test resolving a callable from an allowed path."""
    func = resolve_callable_from_path("os:getcwd", [], allowed_base_paths=["os"])
    import os

    assert func == os.getcwd


def test_resolve_callable_from_path_denied():
    """Test that resolving a callable from a disallowed path fails."""
    with pytest.raises(ImportError, match="is not in the list of allowed base paths"):
        resolve_callable_from_path("sys:getdefaultencoding", [], allowed_base_paths=["os"])


def test_resolve_callable_with_disallowed_schema_path(dummy_module):
    """Test that resolution fails if the signature schema is from a disallowed path."""
    callable_path = "os:access"
    schema_path = "dummy_mod:sig_checker_good"  # Disallowed path
    with pytest.raises(ImportError, match="is not in the list of allowed base paths"):
        resolve_callable_from_path(callable_path, schema_path, allowed_base_paths=["os"])


def test_resolve_callable_with_allowed_schema_path():
    """Test that resolution succeeds if the signature schema is from an allowed path."""
    # We need a checker that actually matches the function signature.
    # Let's check os.access(path, mode, ...)
    mod = types.ModuleType("checker_mod_allowed")

    def access_checker(sig):
        params = list(sig.parameters.keys())
        if "path" not in params or "mode" not in params:
            raise TypeError("Signature mismatch")

    mod.access_checker = access_checker
    sys.modules["checker_mod_allowed"] = mod

    try:
        callable_path = "os:access"
        schema_path = "checker_mod_allowed:access_checker"
        func = resolve_callable_from_path(callable_path, schema_path, allowed_base_paths=["os", "checker_mod_allowed"])
        assert callable(func)
    finally:
        del sys.modules["checker_mod_allowed"]
