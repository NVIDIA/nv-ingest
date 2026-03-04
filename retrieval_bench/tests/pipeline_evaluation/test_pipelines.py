# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for pipeline modules in retrieval_bench.
"""

import pytest

from retrieval_bench.pipelines.backends import VALID_BACKENDS, get_backend_defaults, init_backend


def test_valid_backends_is_non_empty_set():
    assert isinstance(VALID_BACKENDS, set)
    assert VALID_BACKENDS


def test_get_backend_defaults_returns_copy():
    backend = next(iter(VALID_BACKENDS))

    cfg1 = get_backend_defaults(backend)
    cfg2 = get_backend_defaults(backend)

    assert isinstance(cfg1, dict)
    assert isinstance(cfg2, dict)
    assert cfg1 == cfg2

    cfg1["_sentinel"] = 1
    assert "_sentinel" not in cfg2


def test_get_backend_defaults_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_defaults("does-not-exist")


def test_init_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        init_backend(
            "does-not-exist",
            dataset_name="vidore/test",
            corpus_ids=[],
            corpus=[],
        )
