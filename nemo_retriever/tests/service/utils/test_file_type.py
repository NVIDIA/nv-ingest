# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest
from fastapi import UploadFile

from nemo_retriever.common.input_files import INPUT_FORMAT_BY_SUFFIX
from nemo_retriever.service.utils.file_type import FileCategory, FileClassifier


@pytest.mark.parametrize("filename", ["README.MD", "payload.json", "setup.sh"])
def test_documented_plain_text_extensions_are_classified_like_txt(filename: str) -> None:
    upload = UploadFile(filename=filename, file=BytesIO(b"plain text content"))

    classification = FileClassifier.classify(upload)

    assert classification.category == FileCategory.TEXT
    assert classification.content_type == "text/plain"


def test_service_classifier_is_derived_from_common_input_formats() -> None:
    assert FileClassifier.supported_extensions() == frozenset(INPUT_FORMAT_BY_SUFFIX)
    assert {suffix: content_type for suffix, (_, content_type) in FileClassifier.SUFFIX_MAP.items()} == {
        suffix: item.content_type for suffix, item in INPUT_FORMAT_BY_SUFFIX.items()
    }
