# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.internal.schemas.mixins import LowercaseProtocolMixin


### Tests for LowercaseProtocolMixin ###


def test_lowercase_single_protocol_field():
    """Test that a single protocol field is lowercased."""

    class TestSchema(LowercaseProtocolMixin):
        yolox_infer_protocol: str = ""

    config = TestSchema(yolox_infer_protocol="GRPC")
    assert config.yolox_infer_protocol == "grpc"


def test_lowercase_multiple_protocol_fields():
    """Test that multiple protocol fields are all lowercased."""

    class TestSchema(LowercaseProtocolMixin):
        yolox_infer_protocol: str = ""
        ocr_infer_protocol: str = ""
        audio_infer_protocol: str = ""

    config = TestSchema(yolox_infer_protocol="HTTP", ocr_infer_protocol="GRPC", audio_infer_protocol="HtTp")
    assert config.yolox_infer_protocol == "http"
    assert config.ocr_infer_protocol == "grpc"
    assert config.audio_infer_protocol == "http"


def test_mixed_case_normalization():
    """Test that mixed case values are normalized."""

    class TestSchema(LowercaseProtocolMixin):
        test_infer_protocol: str = ""

    config = TestSchema(test_infer_protocol="GrPc")
    assert config.test_infer_protocol == "grpc"


def test_non_protocol_fields_unchanged():
    """Test that non-protocol fields are not affected by the mixin."""

    class TestSchema(LowercaseProtocolMixin):
        yolox_infer_protocol: str = ""
        model_name: str = ""
        timeout: int = 100

    config = TestSchema(yolox_infer_protocol="HTTP", model_name="MyModelNAME", timeout=200)
    assert config.yolox_infer_protocol == "http"  # lowercased
    assert config.model_name == "MyModelNAME"  # unchanged
    assert config.timeout == 200  # unchanged


def test_none_value_handling():
    """Test that None values are handled correctly."""

    class TestSchema(LowercaseProtocolMixin):
        test_infer_protocol: str | None = None

    config = TestSchema(test_infer_protocol=None)
    assert config.test_infer_protocol is None


def test_empty_string_handling():
    """Test that empty strings are handled correctly."""

    class TestSchema(LowercaseProtocolMixin):
        test_infer_protocol: str = ""

    config = TestSchema(test_infer_protocol="")
    assert config.test_infer_protocol == ""


def test_whitespace_stripping():
    """Test that whitespace is stripped from protocol values."""

    class TestSchema(LowercaseProtocolMixin):
        test_infer_protocol: str = ""

    config = TestSchema(test_infer_protocol="  HTTP  ")
    assert config.test_infer_protocol == "http"


def test_protocol_field_name_pattern_matching():
    """Test that only fields ending with '_infer_protocol' are affected."""

    class TestSchema(LowercaseProtocolMixin):
        yolox_infer_protocol: str = ""
        protocol_setting: str = ""  # does NOT end with _infer_protocol
        infer_protocol_prefix: str = ""  # _infer_protocol not at end

    config = TestSchema(
        yolox_infer_protocol="GRPC",
        protocol_setting="UPPERCASE",
        infer_protocol_prefix="ALSOUPPERCAS",
    )
    assert config.yolox_infer_protocol == "grpc"  # lowercased
    assert config.protocol_setting == "UPPERCASE"  # unchanged
    assert config.infer_protocol_prefix == "ALSOUPPERCAS"  # unchanged
