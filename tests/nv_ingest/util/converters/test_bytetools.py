# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest.util.converters.bytetools import base64frombytes
from nv_ingest.util.converters.bytetools import bytesfrombase64
from nv_ingest.util.converters.bytetools import bytesfromhex
from nv_ingest.util.converters.bytetools import hexfrombytes


def test_bytesfromhex_valid_input():
    hex_input = "68656c6c6f"
    expected_output = b"hello"
    assert bytesfromhex(hex_input) == expected_output


def test_hexfrombytes_valid_input():
    bytes_input = b"hello"
    expected_output = "68656c6c6f"
    assert hexfrombytes(bytes_input) == expected_output


def test_bytesfrombase64_valid_input():
    base64_input = "aGVsbG8="
    expected_output = b"hello"
    assert bytesfrombase64(base64_input) == expected_output


def test_base64frombytes_valid_input():
    bytes_input = b"hello"
    expected_output = "aGVsbG8="
    assert base64frombytes(bytes_input) == expected_output


@pytest.mark.parametrize("invalid_input", [123, None, [], {}, True])
def test_bytesfromhex_invalid_input(invalid_input):
    with pytest.raises(TypeError):
        bytesfromhex(invalid_input)


@pytest.mark.parametrize("invalid_input", [123, None, [], {}, True])
def test_hexfrombytes_invalid_input(invalid_input):
    with pytest.raises(AttributeError):
        hexfrombytes(invalid_input)


@pytest.mark.parametrize("invalid_input", [123, None, [], {}, True])
def test_bytesfrombase64_invalid_input(invalid_input):
    with pytest.raises(TypeError):
        bytesfrombase64(invalid_input)


@pytest.mark.parametrize("invalid_input", [123, None, [], {}, True])
def test_base64frombytes_invalid_input(invalid_input):
    with pytest.raises(TypeError):
        base64frombytes(invalid_input)


def test_base64frombytes_with_encoding():
    bytes_input = b"\xf0\xf1\xf2"
    expected_output = "8PHy"
    assert base64frombytes(bytes_input, encoding="utf-8") == expected_output


def test_empty_string_conversions():
    assert bytesfromhex("") == b""
    assert hexfrombytes(b"") == ""
    assert bytesfrombase64("") == b""
    assert base64frombytes(b"") == ""
