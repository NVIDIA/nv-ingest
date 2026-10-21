# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_retriever.common.api.util.string_processing import remove_url_endpoints


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://deplot:8000/v1/chat/completions", "http://deplot:8000"),
        ("http://deplot:8000/v1", "http://deplot:8000"),
        ("http://deplot:8000", "http://deplot:8000"),
        # Hostnames beginning with "v1" must not be truncated.
        ("http://v1-nim:8000/v1/chat/completions", "http://v1-nim:8000"),
        ("http://v1-nim:8000", "http://v1-nim:8000"),
        ("http://v1/v1/embeddings", "http://v1"),
    ],
)
def test_remove_url_endpoints(url: str, expected: str) -> None:
    assert remove_url_endpoints(url) == expected
