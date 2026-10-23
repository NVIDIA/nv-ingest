# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ingest DataFrame transport round-trip helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from nemo_retriever.ingestor.results import (
    concat_ingest_results,
    dataframe_from_transport_records,
    dataframe_to_transport_records,
)
from nemo_retriever.service.services.pipeline_executor import _sanitize_result_data


def test_transport_preserves_column_layout_and_strips_bulky_payloads() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "page_number": [1],
            "text": ["hello"],
            "bytes": [b"pdf-bytes"],
            "_content_type": ["text"],
            "_bbox_xyxy_norm": [[0.1, 0.2, 0.3, 0.4]],
            "_stored_image_uri": ["file:///stored/page.png"],
            "page_image": [{"image_b64": "raw-page", "stored_image_uri": "file:///stored/page.png"}],
            "images": [[{"image_b64": "raw-crop", "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}]],
            "page_elements_v3": [[{"label": "Table"}]],
            "embedding": pd.array([[0.1, 0.2]], dtype=pd.ArrowDtype(pa.list_(pa.float32()))),
            "embedding_error": pd.array([None], dtype=pd.ArrowDtype(pa.string())),
            "metadata": [{"dpi": 200, "source_path": "/a.pdf"}],
        }
    )
    records = dataframe_to_transport_records(df)

    record = records[0]
    assert set(record) == set(df.columns)
    assert record["path"] == "/a.pdf"
    assert record["page_number"] == 1
    assert record["text"] == "hello"
    assert record["bytes"] == "<bytes len=9>"
    assert record["_stored_image_uri"] == "file:///stored/page.png"
    assert record["_bbox_xyxy_norm"] == [0.1, 0.2, 0.3, 0.4]
    assert record["embedding"] is None
    assert record["embedding_error"] is None
    assert record["metadata"] == {"dpi": 200, "source_path": "/a.pdf"}
    assert record["page_image"] == {"image_b64": None, "stored_image_uri": "file:///stored/page.png"}
    assert record["images"][0] == {"image_b64": None, "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}


def test_transport_can_return_legacy_bulk_payloads_when_requested() -> None:
    base64_image = "a" * 600
    embedding = [float(i) for i in range(64)]
    df = pd.DataFrame(
        {
            "page_image": [{"image_b64": base64_image, "stored_image_uri": "file:///stored/page.png"}],
            "images": [[{"image_b64": base64_image}]],
            "embedding": pd.array([embedding], dtype=pd.ArrowDtype(pa.list_(pa.float32()))),
        }
    )

    record = dataframe_to_transport_records(df, return_embeddings=True, return_images=True)[0]

    assert record["page_image"]["image_b64"] == base64_image
    assert record["images"][0]["image_b64"] == base64_image
    assert record["embedding"] == embedding


def test_transport_strips_configured_embedding_column() -> None:
    embedding = np.array([0.5, 0.25], dtype=np.float32)
    df = pd.DataFrame({"text": ["hello"], "custom_embeddings": [embedding]})

    stripped = dataframe_to_transport_records(df, embedding_column="custom_embeddings")[0]
    returned = dataframe_to_transport_records(df, return_embeddings=True, embedding_column="custom_embeddings")[0]

    assert stripped == {"text": "hello", "custom_embeddings": None}
    assert returned == {"text": "hello", "custom_embeddings": [0.5, 0.25]}


def test_transport_preserves_all_legacy_string_values() -> None:
    long_text = "text " * 200
    long_content = "content " * 100
    long_incidental_value = "x" * 600
    df = pd.DataFrame(
        {
            "text": [long_text],
            "metadata": [{"content": long_content, "note": long_incidental_value}],
        }
    )

    record = dataframe_to_transport_records(df)[0]

    assert record["text"] == long_text
    assert record["metadata"]["content"] == long_content
    assert record["metadata"]["note"] == long_incidental_value


def test_transport_preserves_text_across_element_types() -> None:
    long_text = "element text " * 100
    element_types = (
        "text",
        "table",
        "table_caption",
        "chart",
        "chart_caption",
        "infographic",
        "infographic_caption",
        "images",
        "image_caption",
        "audio",
        "video",
        "video_frame",
    )
    df = pd.DataFrame(
        {
            "text": [long_text] * len(element_types),
            "_content_type": list(element_types),
            "custom_embeddings": [[float(index)] for index in range(len(element_types))],
        }
    )

    legacy_records = dataframe_to_transport_records(df)
    compact_records = dataframe_to_transport_records(
        df,
        result_schema="compact",
        return_embeddings=True,
        embedding_column="custom_embeddings",
    )

    assert all(record["text"] == long_text for record in legacy_records)
    assert all(record["text"] == long_text for record in compact_records)
    assert [record["embedding"] for record in compact_records] == [
        [float(index)] for index in range(len(element_types))
    ]


def test_transport_preserves_long_nested_structured_text_collections() -> None:
    long_text = "nested text " * 100
    content_columns = ("table", "chart", "infographic", "images")
    items = [
        {
            "text": long_text,
            "caption": long_text,
            "image_b64": "raw-image",
        }
        for _ in range(21)
    ]
    df = pd.DataFrame({column: [items] for column in content_columns})

    record = dataframe_to_transport_records(df)[0]

    for column in content_columns:
        assert len(record[column]) == 21
        assert record[column][0]["text"] == long_text
        assert record[column][0]["caption"] == long_text
        assert record[column][0]["image_b64"] is None


def test_transport_summarizes_long_lists_after_nested_sanitization() -> None:
    rows = [{"image_b64": f"raw-{idx}", "label": "image"} for idx in range(21)]
    df = pd.DataFrame({"path": ["/a.pdf"], "images": [rows]})

    record = dataframe_to_transport_records(df)[0]

    assert record["images"] == "<list len=21>"


def test_transport_preserves_long_nested_image_lists_when_requested() -> None:
    rows = [{"image_b64": "a" * 600, "label": "image"} for _ in range(21)]
    df = pd.DataFrame({"path": ["/a.pdf"], "images": [rows]})

    record = dataframe_to_transport_records(df, return_images=True)[0]

    assert len(record["images"]) == 21
    assert record["images"][0]["image_b64"] == "a" * 600


def test_transport_compact_document_rows_drop_legacy_metadata_and_bbox() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "page_number": [1],
            "text": ["hello"],
            "_content_type": ["text"],
            "_bbox_xyxy_norm": [[0.1, 0.2, 0.3, 0.4]],
            "_stored_image_uri": ["file:///stored/page.png"],
            "page_image": [{"image_b64": "raw-page", "stored_image_uri": "file:///stored/page.png"}],
            "images": [[{"image_b64": "raw-crop", "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}]],
            "embedding": [[0.3, 0.4]],
            "metadata": [{"dpi": 200, "source_path": "/a.pdf"}],
        }
    )

    assert dataframe_to_transport_records(df, result_schema="compact") == [
        {
            "text": "hello",
            "source_id": "/a.pdf",
            "element_type": "text",
            "page_number": 1,
            "stored_image_uri": "file:///stored/page.png",
        }
    ]


def test_transport_compact_media_rows_return_timings() -> None:
    df = pd.DataFrame(
        {
            "path": ["/tmp/chunk-000.wav"],
            "source_path": ["/media/call.wav"],
            "page_number": [3],
            "text": ["hello from audio"],
            "_content_type": ["audio"],
            "metadata": [
                {
                    "duration": 30.0,
                    "segment_start_seconds": 10.5,
                    "segment_end_seconds": 12.0,
                    "source_path": "/media/call.wav",
                }
            ],
            "embedding": [[0.1, 0.2]],
        }
    )

    assert dataframe_to_transport_records(df, result_schema="compact") == [
        {
            "text": "hello from audio",
            "source_id": "/media/call.wav",
            "element_type": "audio",
            "start_time_seconds": 10.5,
            "end_time_seconds": 12.0,
            "duration_seconds": 1.5,
        }
    ]


def test_transport_compact_rows_ignore_metadata_embedding() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "text": ["hello"],
            "metadata": [{"embedding": [0.1, 0.2]}],
        }
    )

    assert dataframe_to_transport_records(df, result_schema="compact", return_embeddings=True) == [
        {"text": "hello", "source_id": "/a.pdf", "element_type": "text"}
    ]


def test_transport_compact_rows_return_arrow_embedding_when_requested() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf", "/b.pdf"],
            "text": ["hello", "unembedded"],
            "embedding": pd.array([[0.5, 0.25], None], dtype=pd.ArrowDtype(pa.list_(pa.float32()))),
        }
    )

    records = dataframe_to_transport_records(df, result_schema="compact", return_embeddings=True)

    assert records[0]["embedding"] == [0.5, 0.25]
    assert "embedding" not in records[1]


def test_transport_compact_rows_return_configured_column_embedding_when_requested() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf"],
            "text": ["hello"],
            "embedding": [[9.0, 9.0]],
            "custom_embeddings": [[0.5, 0.6]],
        }
    )

    record = dataframe_to_transport_records(
        df, result_schema="compact", return_embeddings=True, embedding_column="custom_embeddings"
    )[0]

    assert record["embedding"] == [0.5, 0.6]


def test_round_trip_matches_inprocess_column_layout() -> None:
    df = pd.DataFrame(
        {
            "path": ["/a.pdf", "/a.pdf"],
            "page_number": [1, 2],
            "text": ["a", "b"],
            "metadata": [{"content_metadata": {"type": "text"}}, {"content_metadata": {"type": "text"}}],
        }
    )
    rebuilt = dataframe_from_transport_records(dataframe_to_transport_records(df))
    assert list(rebuilt.columns) == list(df.columns)
    assert len(rebuilt) == len(df)
    assert rebuilt["text"].tolist() == df["text"].tolist()


def test_sanitize_result_data_delegates_to_shared_helper() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "text": ["x"]})
    assert _sanitize_result_data(df) == dataframe_to_transport_records(df)


def test_sanitize_result_data_accepts_compact_schema() -> None:
    df = pd.DataFrame({"path": ["/x.pdf"], "text": ["x"]})
    assert _sanitize_result_data(df, result_schema="compact") == dataframe_to_transport_records(
        df,
        result_schema="compact",
    )


def test_sanitize_result_data_forwards_compact_embedding_flag() -> None:
    df = pd.DataFrame(
        {
            "text": ["x"],
            "embedding": [[9.0]],
            "custom_embeddings": [[0.1]],
        }
    )

    assert _sanitize_result_data(
        df, result_schema="compact", return_embeddings=True, embedding_column="custom_embeddings"
    ) == dataframe_to_transport_records(
        df, result_schema="compact", return_embeddings=True, embedding_column="custom_embeddings"
    )


def test_sanitize_result_data_forwards_legacy_payload_flags() -> None:
    df = pd.DataFrame({"embedding": [[0.1]], "page_image": [{"image_b64": "raw"}]})

    assert _sanitize_result_data(df, return_embeddings=True, return_images=True) == dataframe_to_transport_records(
        df,
        return_embeddings=True,
        return_images=True,
    )


def test_concat_ingest_results_follows_document_order() -> None:
    rows_a = [{"path": "/a.pdf", "page_number": 1, "text": "a"}]
    rows_b = [{"path": "/b.pdf", "page_number": 1, "text": "b"}]
    combined = concat_ingest_results(
        {"doc-b": rows_b, "doc-a": rows_a},
        ["doc-a", "doc-b"],
    )
    assert combined["path"].tolist() == ["/a.pdf", "/b.pdf"]
    assert list(combined.columns) == ["path", "page_number", "text"]
