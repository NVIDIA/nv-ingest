# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

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
from nemo_retriever.operators.embed.text_embed import TextEmbedGPUActor, embed_text_1b_v2


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


def _assert_embedding_types(result: pd.DataFrame) -> None:
    assert result["embedding"].dtype == EMBEDDING_DTYPE
    assert result["embedding"].dtype.pyarrow_dtype.value_type == pa.float32()
    assert result["embedding_error"].dtype == EMBEDDING_ERROR_DTYPE


def test_create_text_embeddings_for_df_writes_float32_column_without_metadata_embedding() -> None:
    source = pd.DataFrame({"text": ["one"], "metadata": [{"source_path": "one.txt"}]})
    values = [0.123456789, -1.234567891]

    result, _ = create_text_embeddings_for_df(
        source,
        task_config={"embedder": lambda texts: [values for _ in texts]},
    )

    _assert_embedding_types(result)
    expected = np.asarray(values, dtype=np.float32)
    actual = np.asarray(result.loc[0, "embedding"], dtype=np.float32)
    assert actual.tobytes() == expected.tobytes()
    assert pd.isna(result.loc[0, "embedding_error"])
    assert "embedding" not in result.loc[0, "metadata"]
    assert "info_message_metadata" not in result.loc[0, "metadata"]


def test_create_text_embeddings_for_df_marks_missing_content_as_null() -> None:
    result, _ = create_text_embeddings_for_df(
        pd.DataFrame({"text": [None], "metadata": [{}]}),
        task_config={"embedder": lambda _texts: []},
    )

    _assert_embedding_types(result)
    assert pd.isna(result.loc[0, "embedding"])
    assert pd.isna(result.loc[0, "embedding_error"])


@pytest.mark.parametrize(
    "source",
    [
        pd.DataFrame({"text": [None, None]}),
        pd.DataFrame({"text": pd.Series([], dtype=object)}),
    ],
)
def test_create_text_embeddings_for_df_keeps_explicit_types_for_null_and_empty_batches(source: pd.DataFrame) -> None:
    result, _ = create_text_embeddings_for_df(source, task_config={"embedder": lambda _texts: []})

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].isna().all()


def test_embed_text_main_text_embed_records_one_line_batch_error_and_counts() -> None:
    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one"]}),
        model=_FixedEmbedder(error=RuntimeError("boom\nline2")),
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].tolist() == [
        "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause"
    ]
    assert not any("boom" in error for error in result["embedding_error"])
    assert result["embedding_v1_counts_by_label"].tolist() == [{"unembedded": 1, "failed": 1}]
    assert result["embedding_v1_num_detections"].tolist() == [0]


def test_embed_text_main_text_embed_counts_successful_embedding() -> None:
    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one"]}),
        model=_FixedEmbedder(output=[[0.1, 0.2]]),
    )

    _assert_embedding_types(result)
    assert result["embedding"].notna().all()
    assert result["embedding_error"].isna().all()
    assert result["embedding_v1_counts_by_label"].tolist() == [{"embedded": 1}]
    assert result["embedding_v1_num_detections"].tolist() == [1]


def test_embed_text_main_text_embed_counts_empty_vector_as_unembedded() -> None:
    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one"]}),
        model=_FixedEmbedder(output=[[]]),
    )

    _assert_embedding_types(result)
    assert result["embedding"].array.__arrow_array__().to_pylist() == [None]
    assert result["embedding_error"].isna().all()
    assert result["embedding_v1_counts_by_label"].tolist() == [{"unembedded": 1}]
    assert result["embedding_v1_num_detections"].tolist() == [0]


def test_embed_text_main_text_embed_preserves_successful_vector_with_empty_neighbor() -> None:
    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one", "two"]}),
        model=_FixedEmbedder(output=[[0.1, 0.2], []]),
    )

    _assert_embedding_types(result)
    np.testing.assert_array_equal(
        np.asarray(result.loc[0, "embedding"], dtype=np.float32),
        np.asarray([0.1, 0.2], dtype=np.float32),
    )
    assert result["embedding"].array.__arrow_array__().to_pylist()[1] is None
    assert result["embedding_error"].array.__arrow_array__().to_pylist() == [None, None]


def test_assign_embedding_columns_masks_empty_vectors_and_preserves_errors() -> None:
    source = pd.DataFrame({"text": ["one", "two", "three"]})

    result = assign_embedding_columns(
        source,
        [[1.0, 2.0], [], None],
        [None, None, "e"],
        output_column="embedding",
    )

    _assert_embedding_types(result)
    assert result["embedding"].array.__arrow_array__().to_pylist() == [[1.0, 2.0], None, None]
    assert result["embedding_error"].array.__arrow_array__().to_pylist() == [None, None, "e"]
    assert source.columns.tolist() == ["text"]


def test_assign_embedding_columns_keeps_typed_columns_for_all_empty_vectors() -> None:
    result = assign_embedding_columns(
        pd.DataFrame({"text": ["one", "two"]}),
        [[], []],
        [None, None],
        output_column="embedding",
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].isna().all()


@pytest.mark.parametrize(
    "output",
    [
        [[0.1, 0.2], [0.3, 0.4]],
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    ],
)
def test_embed_text_1b_v2_writes_float32_vectors(output) -> None:
    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["one", "two"]}),
        model=_FixedEmbedder(output=output),
    )

    _assert_embedding_types(result)
    assert result["embedding"].notna().all()
    assert result["embedding_error"].isna().all()


def test_embed_text_1b_v2_records_chunk_failure() -> None:
    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["one", "two"]}),
        model=_FixedEmbedder(error=RuntimeError("chunk\nfailed")),
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].tolist() == ["embed: RuntimeError: chunk failed"] * 2


def test_embed_text_1b_v2_records_text_extraction_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.operators.embed.text_embed._text_from_row",
        Mock(side_effect=RuntimeError("bad\ntext")),
    )

    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["one"]}),
        model=_FixedEmbedder(output=[]),
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].tolist() == ["extract_text: RuntimeError: bad text"]


def test_embed_text_1b_v2_keeps_types_for_empty_text_batch() -> None:
    result = embed_text_1b_v2(
        pd.DataFrame({"text": ["", "   "]}),
        model=_FixedEmbedder(output=[]),
    )

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].isna().all()


def test_text_embed_gpu_actor_failure_returns_typed_error_columns(monkeypatch) -> None:
    actor = object.__new__(TextEmbedGPUActor)
    actor.detect_kwargs = {}
    monkeypatch.setattr(actor, "run", Mock(side_effect=RuntimeError("actor\nfailed")), raising=False)

    result = actor(pd.DataFrame({"text": ["one"]}))

    _assert_embedding_types(result)
    assert result["embedding"].isna().all()
    assert result["embedding_error"].tolist() == ["actor_call: RuntimeError: actor failed"]


def test_assign_embedding_columns_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        assign_embedding_columns(pd.DataFrame({"text": ["one"]}), [], [], output_column="embedding")
