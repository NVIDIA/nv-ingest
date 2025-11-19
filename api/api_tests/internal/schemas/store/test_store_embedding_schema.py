# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema


def test_embedding_storage_schema_defaults():
    schema = EmbeddingStorageSchema()
    assert schema.raise_on_failure is False


def test_embedding_storage_schema_accepts_true():
    schema = EmbeddingStorageSchema(raise_on_failure=True)
    assert schema.raise_on_failure is True


def test_embedding_storage_schema_accepts_false():
    schema = EmbeddingStorageSchema(raise_on_failure=False)
    assert schema.raise_on_failure is False


def test_embedding_storage_schema_accepts_bool_like_values():
    # Regular bool field accepts truthy/falsy values
    schema = EmbeddingStorageSchema(raise_on_failure=1)
    assert schema.raise_on_failure is True

    schema = EmbeddingStorageSchema(raise_on_failure=0)
    assert schema.raise_on_failure is False

    schema = EmbeddingStorageSchema(raise_on_failure="True")
    assert schema.raise_on_failure is True


def test_embedding_storage_schema_rejects_extra_fields():
    with pytest.raises(ValidationError) as excinfo:
        EmbeddingStorageSchema(raise_on_failure=True, extra_field="oops")
    assert "Extra inputs are not permitted" in str(excinfo.value)
