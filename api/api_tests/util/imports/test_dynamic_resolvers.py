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
