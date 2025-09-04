# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import OrderedDict

import pytest

from nv_ingest_api.util.logging.sanitize import sanitize_for_logging


def test_basic_redaction_dict():
    data = {
        "username": "alice",
        "password": "p@ssw0rd",
        "api_key": "abc123",
        "nested": {"secret": "shh", "ok": 1},
    }
    out = sanitize_for_logging(data)

    assert out["username"] == "alice"
    assert out["password"] == "***REDACTED***"
    assert out["api_key"] == "***REDACTED***"
    assert out["nested"]["secret"] == "***REDACTED***"
    assert out["nested"]["ok"] == 1


def test_case_insensitive_and_hyphen_keys():
    data = {
        "Authorization": "Bearer token",
        "PASSWORD": "x",
        "X-API-Key": "k",
        "x-api-key": "k2",
    }
    out = sanitize_for_logging(data)

    assert out["Authorization"] == "***REDACTED***"
    assert out["PASSWORD"] == "***REDACTED***"
    assert out["X-API-Key"] == "***REDACTED***"
    assert out["x-api-key"] == "***REDACTED***"


def test_partial_key_names_not_redacted():
    data = {
        "password_hint": "blue",
        "client_secret_hint": "maybe",
        "auth_token_backup": "nope",
    }
    out = sanitize_for_logging(data)

    assert out["password_hint"] == "blue"
    assert out["client_secret_hint"] == "maybe"
    assert out["auth_token_backup"] == "nope"


def test_nested_structures_list_of_dicts():
    data = {
        "items": [
            {"password": "a"},
            {"ok": 1, "children": [{"secret": "b"}]},
        ],
        "tuples": ({"client_secret": "c"}, {"ok": 2}),
    }
    out = sanitize_for_logging(data)

    assert out["items"][0]["password"] == "***REDACTED***"
    assert out["items"][1]["ok"] == 1
    assert out["items"][1]["children"][0]["secret"] == "***REDACTED***"
    assert out["tuples"][0]["client_secret"] == "***REDACTED***"
    assert out["tuples"][1]["ok"] == 2


def test_sequence_types_preserved():
    data_list = [
        {"access_token": "tok"},
        "string",
        b"bytes",
    ]
    out_list = sanitize_for_logging(data_list)

    assert isinstance(out_list, list)
    assert out_list[0]["access_token"] == "***REDACTED***"
    assert out_list[1] == "string"
    assert out_list[2] == b"bytes"

    data_tuple = ({"refresh_token": "tok"}, 123)
    out_tuple = sanitize_for_logging(data_tuple)
    assert isinstance(out_tuple, tuple)
    assert out_tuple[0]["refresh_token"] == "***REDACTED***"
    assert out_tuple[1] == 123


def test_custom_sensitive_keys_and_redaction():
    data = {"password": "x", "token": "y"}
    out = sanitize_for_logging(data, sensitive_keys={"token"}, redaction="XXX")

    # Only custom set is honored
    assert out["password"] == "x"
    assert out["token"] == "XXX"


def test_input_not_mutated_and_deep_copied():
    data = {"outer": {"password": "x", "ok": 1}}
    data_copy = copy.deepcopy(data)
    out = sanitize_for_logging(data)

    assert data == data_copy, "Input should not be mutated"
    assert out is not data
    assert out["outer"] is not data["outer"], "Nested mapping should be a new object"
    assert out["outer"]["password"] == "***REDACTED***"


def test_mapping_type_preserved_for_ordereddict():
    od = OrderedDict()
    od["password"] = "x"
    od["ok"] = 1
    out = sanitize_for_logging(od)

    assert isinstance(out, OrderedDict)
    assert out["password"] == "***REDACTED***"
    assert out["ok"] == 1


def test_non_mapping_non_sequence_returned_as_is():
    # set is not a Mapping or Sequence; it should be returned as-is
    s = {1, 2, 3}
    out = sanitize_for_logging(s)

    assert out == s


def test_strings_and_bytes_unchanged():
    assert sanitize_for_logging("secret") == "secret"
    raw = b"binary"
    assert sanitize_for_logging(raw) is raw


def test_top_level_list_of_dicts():
    data = [
        {"username": "bob", "authorization": "abc"},
        {"x-api-key": "k"},
    ]
    out = sanitize_for_logging(data)

    assert out[0]["username"] == "bob"
    assert out[0]["authorization"] == "***REDACTED***"
    assert out[1]["x-api-key"] == "***REDACTED***"


def test_handles_missing_or_weird_types_gracefully():
    class Weird:
        __slots__ = ("a",)

        def __init__(self):
            self.a = 1

    w = Weird()
    # Should just return as-is
    assert sanitize_for_logging(w) is w


@pytest.mark.parametrize(
    "key",
    [
        "access_token",
        "api_key",
        "API_KEY",
        "authorization",
        "Authorization",
        "auth_token",
        "client_secret",
        "hf_access_token",
        "hugging_face_access_token",
        "password",
        "refresh_token",
        "secret",
        "ssl_cert",
        "x-api-key",
    ],
)
def test_default_sensitive_keys_cover_common_cases(key):
    data = {key: "value"}
    out = sanitize_for_logging(data)
    assert list(out.values())[0] == "***REDACTED***"


def test_pydantic_model_redaction():
    pydantic = pytest.importorskip("pydantic")

    class Nested(pydantic.BaseModel):
        client_secret: str
        other: int

    class Model(pydantic.BaseModel):
        username: str
        password: str
        nested: Nested

    m = Model(username="u", password="p", nested=Nested(client_secret="c", other=5))
    out = sanitize_for_logging(m)

    # Expect a dict back with redacted sensitive fields
    assert isinstance(out, dict)
    assert out["username"] == "u"
    assert out["password"] == "***REDACTED***"
    assert out["nested"]["client_secret"] == "***REDACTED***"
    assert out["nested"]["other"] == 5


def test_pydantic_models_in_sequence():
    pydantic = pytest.importorskip("pydantic")

    class Token(pydantic.BaseModel):
        access_token: str

    seq = [Token(access_token="abc"), {"refresh_token": "r"}]
    out = sanitize_for_logging(seq)

    assert isinstance(out, list)
    assert out[0]["access_token"] == "***REDACTED***"
    assert out[1]["refresh_token"] == "***REDACTED***"
