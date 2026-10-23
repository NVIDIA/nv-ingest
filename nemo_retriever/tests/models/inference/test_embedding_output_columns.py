# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from nemo_retriever.models.inference.main_text_embed import (
    EMBEDDING_DTYPE,
    EMBEDDING_ERROR_DTYPE,
    assign_embedding_columns,
    create_text_embeddings_for_df,
)
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
from nemo_retriever.operators.embed.text_embed import embed_text_1b_v2


class _FixedEmbedder:
    def __init__(self, output=None, error: BaseException | None = None) -> None:
        self.output = output
        self.error = error

    def embed(self, texts, *, batch_size):
        if self.error is not None:
            raise self.error
        if callable(self.output):
            return self.output(texts)
        return self.output


def _assert_embedding_types(result: pd.DataFrame, output_column: str = "embedding") -> None:
    assert result[output_column].dtype == EMBEDDING_DTYPE
    assert result[output_column].dtype.pyarrow_dtype.value_type == pa.float32()
    assert result[f"{output_column}_error"].dtype == EMBEDDING_ERROR_DTYPE


def test_create_text_embeddings_for_df_writes_typed_columns() -> None:
    source = pd.DataFrame({"text": ["one", None], "metadata": [{"source_path": "one.txt"}, {}]})
    values = [0.123456789, -1.234567891]

    result, _ = create_text_embeddings_for_df(
        source,
        task_config={"embedder": lambda texts: [values for _ in texts]},
    )

    _assert_embedding_types(result)
    expected = np.asarray(values, dtype=np.float32)
    actual = np.asarray(result.loc[0, "embedding"], dtype=np.float32)
    assert actual.tobytes() == expected.tobytes()
    assert result["embedding_error"].isna().all()
    assert "embedding" not in result.loc[0, "metadata"]
    assert pd.isna(result.loc[1, "embedding"])

    empty, _ = create_text_embeddings_for_df(
        pd.DataFrame({"text": pd.Series([], dtype=object)}),
        task_config={"embedder": lambda _texts: []},
    )
    _assert_embedding_types(empty)
    assert empty.empty


def test_embed_text_main_text_embed_writes_null_for_empty_and_missing_vectors() -> None:
    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one", "two", "three"]}),
        model=_FixedEmbedder(output=[[0.1, 0.2], [], None]),
    )

    _assert_embedding_types(result)
    vectors = result["embedding"].array.__arrow_array__().to_pylist()
    np.testing.assert_array_equal(
        np.asarray(vectors[0], dtype=np.float32),
        np.asarray([0.1, 0.2], dtype=np.float32),
    )
    assert vectors[1:] == [None, None]
    assert result["embedding_error"].isna().all()


def test_embed_text_1b_v2_writes_configured_output_column() -> None:
    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["one"]}),
        model=_FixedEmbedder(output=[[0.1, 0.2]]),
        output_column="custom_embedding",
    )

    _assert_embedding_types(result, "custom_embedding")
    np.testing.assert_array_equal(
        np.asarray(result.loc[0, "custom_embedding"], dtype=np.float32),
        np.asarray([0.1, 0.2], dtype=np.float32),
    )
    assert result["custom_embedding_error"].isna().all()
    assert "embedding" not in result.columns


def test_embed_text_1b_v2_records_chunk_failure() -> None:
    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["one", "two"]}),
        model=_FixedEmbedder(error=RuntimeError("chunk\nfailed")),
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].tolist() == ["embed: RuntimeError: chunk failed"] * 2


def test_assign_embedding_columns_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        assign_embedding_columns(pd.DataFrame({"text": ["one"]}), [], [], output_column="embedding")


def test_assign_embedding_columns_keeps_each_vector_length() -> None:
    frame = pd.DataFrame({"text": ["a", "b"]})

    result = assign_embedding_columns(frame, [[1.0, 2.0], [3.0, 4.0, 5.0]], [None, None], output_column="embedding")

    _assert_embedding_types(result)
    first, second = result["embedding"].tolist()
    assert list(first) == [1.0, 2.0]
    assert list(second) == [3.0, 4.0, 5.0]
