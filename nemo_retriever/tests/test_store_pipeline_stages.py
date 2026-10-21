# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-level tests for StoreOperator."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
from nemo_retriever.common.modality.content_transforms import explode_content_to_rows
from nemo_retriever.common.params import StoreParams
from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.graph import InprocessExecutor, StoreOperator, UDFOperator
from PIL import Image


def _make_tiny_png_b64(width: int = 4, height: int = 4, color=(255, 0, 0)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=color)
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_embedded_df(b64: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": "/docs/test.pdf",
                "page_number": 1,
                "text": "Sample page text",
                "_content_type": "text",
                "_bbox_xyxy_norm": None,
                "_image_b64": b64,
                "page_image": {"image_b64": b64, "stored_image_uri": "file:///old/page.png"},
                "table": [
                    {
                        "text": "col1|col2",
                        "image_b64": b64,
                        "stored_image_uri": "file:///old/table.png",
                    }
                ],
            }
        ]
    )


class TestStoreOperatorInGraph:
    def test_store_operator_writes_row_image_and_sets_top_level_uri(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)

        graph = UDFOperator(lambda x: x, name="Identity") >> StoreOperator(
            params=StoreParams(storage_uri=str(tmp_path))
        )
        result = InprocessExecutor(graph, show_progress=False).ingest(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert stored_uri.startswith("file://")
        assert Path(urlparse(stored_uri).path).exists()

    @pytest.mark.parametrize("metadata_key", ["image_metadata", "table_metadata", "chart_metadata"])
    def test_store_operator_exposes_uploaded_image_uri_in_structured_metadata(self, tmp_path: Path, metadata_key: str):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)
        df.at[0, "_content_type"] = metadata_key.removesuffix("_metadata")
        df["metadata"] = [
            {
                "content_metadata": {"type": "structured"},
                metadata_key: {"table_format": "image", "uploaded_image_uri": ""},
            }
        ]

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert result.iloc[0]["metadata"]["content_metadata"]["uploaded_image_uri"] == stored_uri
        assert result.iloc[0]["metadata"][metadata_key]["uploaded_image_uri"] == stored_uri

    def test_store_operator_synchronizes_existing_stored_uri_into_public_metadata(self, tmp_path: Path):
        df = _make_embedded_df(None)
        df["_stored_image_uri"] = ["file:///stored/table.png"]
        df["metadata"] = [
            {
                "table_metadata": {"table_format": "image", "uploaded_image_uri": ""},
            }
        ]

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert result.iloc[0]["metadata"]["table_metadata"]["uploaded_image_uri"] == "file:///stored/table.png"

    def test_store_operator_does_not_publish_inherited_page_uri_as_table_asset(self, tmp_path: Path):
        df = _make_embedded_df(None)
        df["_content_type"] = ["table"]
        df["_stored_image_uri"] = ["file:///stored/page.png"]
        df["page_image"] = [{"image_b64": None, "stored_image_uri": "file:///stored/page.png"}]
        df["metadata"] = [
            {
                "content_metadata": {"type": "structured", "uploaded_image_uri": ""},
                "table_metadata": {
                    "table_format": "image",
                    "uploaded_image_uri": "file:///stored/table.png",
                },
            }
        ]

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert result.iloc[0]["metadata"]["content_metadata"]["uploaded_image_uri"] == ""
        assert result.iloc[0]["metadata"]["table_metadata"]["uploaded_image_uri"] == "file:///stored/table.png"

    def test_store_operator_clears_row_and_page_payloads_after_write(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert result.iloc[0]["_image_b64"] is None
        assert result.iloc[0]["page_image"]["image_b64"] is None
        assert result.iloc[0]["page_image"]["stored_image_uri"] == result.iloc[0]["_stored_image_uri"]
        assert result.iloc[0]["table"][0]["image_b64"] is None
        assert result.iloc[0]["table"][0]["stored_image_uri"] == "file:///old/table.png"

    def test_store_operator_writes_nested_image_payloads(self, tmp_path: Path):
        b64 = _make_tiny_png_b64(color=(0, 255, 0))
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "images": [
                        {
                            "text": "figure",
                            "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4],
                            "image_b64": b64,
                        }
                    ],
                }
            ]
        )

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)
        nested = result.iloc[0]["images"][0]
        assert nested["image_b64"] is None
        assert nested["stored_image_uri"].startswith("file://")
        assert Path(urlparse(nested["stored_image_uri"]).path).exists()
        assert "_stored_image_uri" not in result.columns

    def test_store_operator_names_nested_images_from_source_and_chunk(self, tmp_path: Path):
        first_b64 = _make_tiny_png_b64(color=(0, 255, 0))
        second_b64 = _make_tiny_png_b64(color=(0, 0, 255))
        df = pd.DataFrame(
            [
                {
                    "path": "/docs/quarterly report.pdf",
                    "page_number": 2,
                    "images": [
                        {"image_b64": first_b64},
                        {"image_b64": second_b64},
                    ],
                }
            ]
        )

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        image_uris = [item["stored_image_uri"] for item in result.iloc[0]["images"]]
        image_names = [Path(urlparse(uri).path).name for uri in image_uris]
        assert image_names[0].startswith("quarterly_report-p2-image-c1-")
        assert image_names[1].startswith("quarterly_report-p2-image-c2-")
        assert image_names[0].endswith(".png")
        assert image_names[1].endswith(".png")
        assert image_names[0] != image_names[1]
        assert Path(urlparse(image_uris[0]).path).read_bytes() == base64.b64decode(first_b64)
        assert Path(urlparse(image_uris[1]).path).read_bytes() == base64.b64decode(second_b64)

    def test_store_operator_does_not_overwrite_page_uri_for_element_rows(self, tmp_path: Path):
        page_b64 = _make_tiny_png_b64(color=(255, 0, 0))
        element_b64 = _make_tiny_png_b64(color=(0, 0, 255))
        df = _make_embedded_df(element_b64)
        df.at[0, "_content_type"] = "images"
        df.at[0, "page_image"] = {"image_b64": page_b64, "stored_image_uri": "file:///old/page.png"}

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert Path(urlparse(stored_uri).path).read_bytes() == base64.b64decode(element_b64)
        assert result.iloc[0]["_image_b64"] is None
        assert result.iloc[0]["page_image"]["image_b64"] is None
        assert result.iloc[0]["page_image"]["stored_image_uri"] == "file:///old/page.png"

    def test_store_operator_associates_text_mode_structured_row_with_page_crop(self, tmp_path: Path):
        page_b64 = _make_tiny_png_b64(width=100, height=80)
        source = pd.DataFrame(
            [
                {
                    "path": "/docs/test.pdf",
                    "page_number": 1,
                    "text": "Page text",
                    "page_image": {"image_b64": page_b64},
                    "table": [
                        {
                            "text": "Table text",
                            "bbox_xyxy_norm": [0.25, 0.25, 0.75, 0.75],
                        }
                    ],
                    "metadata": {
                        "content_metadata": {"uploaded_image_uri": ""},
                        "table_metadata": {"uploaded_image_uri": ""},
                    },
                }
            ]
        )
        exploded = explode_content_to_rows(source, modality="text")
        assert "_image_b64" not in exploded.columns

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(exploded)

        page_row = result.loc[result["_content_type"] == "text"].iloc[0]
        table_row = result.loc[result["_content_type"] == "table"].iloc[0]
        page_uri = page_row["_stored_image_uri"]
        table_uri = table_row["_stored_image_uri"]

        assert table_uri != page_uri
        assert Image.open(Path(urlparse(page_uri).path)).size == (100, 80)
        assert Image.open(Path(urlparse(table_uri).path)).size == (50, 40)
        assert table_row["page_image"]["stored_image_uri"] == page_uri
        assert table_row["metadata"]["content_metadata"]["uploaded_image_uri"] == table_uri
        assert table_row["metadata"]["table_metadata"]["uploaded_image_uri"] == table_uri

        result["metadata"] = result["metadata"].apply(lambda metadata: {**metadata, "embedding": [0.1, 0.2]})
        records = to_client_vdb_records(result)[0]
        table_record = next(
            record for record in records if record["metadata"]["content_metadata"].get("type") == "table"
        )
        content_metadata = table_record["metadata"]["content_metadata"]
        assert content_metadata["stored_image_uri"] == table_uri
        assert content_metadata["uploaded_image_uri"] == table_uri
        assert content_metadata["bbox_xyxy_norm"] == [0.25, 0.25, 0.75, 0.75]

    def test_store_operator_crops_structured_row_with_numpy_bbox(self, tmp_path: Path):
        page_b64 = _make_tiny_png_b64(width=100, height=80)
        df = pd.DataFrame(
            [
                {
                    "_content_type": "table",
                    "_bbox_xyxy_norm": np.array([0.25, 0.25, 0.75, 0.75]),
                    "page_image": {"image_b64": page_b64},
                }
            ]
        )

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert Image.open(Path(urlparse(stored_uri).path)).size == (50, 40)

    def test_store_operator_skips_rows_without_image_b64(self, tmp_path: Path):
        df = _make_embedded_df(None)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        assert not list(tmp_path.rglob("*"))
        assert "_stored_image_uri" not in result.columns

    def test_store_operator_uses_page_image_when_row_image_column_is_absent(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64).drop(columns=["_image_b64"])

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)
        assert result.iloc[0]["_stored_image_uri"].startswith("file://")
        assert result.iloc[0]["page_image"]["image_b64"] is None
        assert result.iloc[0]["page_image"]["stored_image_uri"] == result.iloc[0]["_stored_image_uri"]

    def test_store_operator_writes_top_level_image_b64_video_frame(self, tmp_path: Path):
        b64 = _make_tiny_png_b64()
        df = pd.DataFrame(
            [
                {
                    "path": "/media/demo.mp4",
                    "_content_type": "video_frame",
                    "image_b64": b64,
                }
            ]
        )

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)
        assert result.iloc[0]["_stored_image_uri"].startswith("file://")
        assert result.iloc[0]["image_b64"] is None

    def test_store_operator_writes_uri_into_null_typed_arrow_column(self, tmp_path: Path):
        pa = pytest.importorskip("pyarrow")

        b64 = _make_tiny_png_b64()
        arrow_table = pa.table(
            {
                "page_image": pa.array([{"image_b64": b64}], type=pa.struct([("image_b64", pa.string())])),
                "_content_type": pa.array(["text"], type=pa.string()),
                # Upstream content transforms emit an all-None column, which Arrow types as null.
                "_stored_image_uri": pa.array([None], type=pa.null()),
            }
        )
        df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
        assert isinstance(df["_stored_image_uri"].dtype, pd.ArrowDtype)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        stored_uri = result.iloc[0]["_stored_image_uri"]
        assert isinstance(stored_uri, str)
        assert stored_uri.startswith("file://")
        assert Path(urlparse(stored_uri).path).read_bytes() == base64.b64decode(b64)
        assert result.iloc[0]["page_image"]["stored_image_uri"] == stored_uri
        list(result.iterrows())
        pa.Table.from_pandas(result)

    def test_store_operator_stores_numpy_backed_element_images(self, tmp_path: Path):
        pa = pytest.importorskip("pyarrow")

        element_b64 = _make_tiny_png_b64(color=(0, 0, 255))
        arrow_table = pa.table(
            {
                "images": pa.array(
                    [[{"text": "fig", "image_b64": element_b64}]],
                    type=pa.list_(pa.struct([("text", pa.string()), ("image_b64", pa.string())])),
                ),
                "_content_type": pa.array(["images"], type=pa.string()),
            }
        )
        # Ray Data hands batches over with list columns converted to NumPy arrays.
        df = arrow_table.to_pandas()
        assert isinstance(df.iloc[0]["images"], np.ndarray)

        result = StoreOperator(params=StoreParams(storage_uri=str(tmp_path))).process(df)

        element = result.iloc[0]["images"][0]
        assert element["image_b64"] is None
        assert Path(urlparse(element["stored_image_uri"]).path).read_bytes() == base64.b64decode(element_b64)
        list(result.iterrows())
        pa.Table.from_pandas(result)

    def test_store_operator_updates_arrow_backed_page_image_struct(self, tmp_path: Path):
        pa = pytest.importorskip("pyarrow")

        b64 = _make_tiny_png_b64()
        struct_type = pa.struct(
            [
                ("image_b64", pa.string()),
                ("encoding", pa.string()),
                ("orig_shape_hw", pa.list_(pa.int64())),
            ]
        )
        page_image = {"image_b64": b64, "encoding": "png", "orig_shape_hw": [4, 4]}
        arr = pa.array([page_image], type=struct_type)
        df = pd.DataFrame(
            {
                "page_image": pd.arrays.ArrowExtensionArray(arr),
                "_content_type": ["text"],
            }
        )

        result = StoreOperator(
            params=StoreParams(storage_uri=str(tmp_path), strip_base64=True, storage_options={"auto_mkdir": True})
        ).process(df)

        files = list(tmp_path.rglob("*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == base64.b64decode(b64)
        assert result.iloc[0]["page_image"]["image_b64"] is None
        assert result.iloc[0]["page_image"]["stored_image_uri"] == result.iloc[0]["_stored_image_uri"]
        assert result.iloc[0]["page_image"]["encoding"] == "png"

    def test_store_operator_does_not_rewrite_page_image_when_strip_false(self, monkeypatch):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64).drop(columns=["_image_b64"])
        calls: list[str] = []

        class _Writer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, data: bytes) -> int:
                return len(data)

        def _fake_open(path: str, mode: str = "rb", **kwargs):
            calls.append(path)
            return _Writer()

        monkeypatch.setattr("nemo_retriever.operators.graph_ops.store_operator.fsspec.open", _fake_open)

        result = StoreOperator(params=StoreParams(storage_uri="memory://stored", strip_base64=False)).process(df)

        assert len(calls) == 1
        assert result.iloc[0]["_stored_image_uri"].startswith("memory://stored/")
        assert result.iloc[0]["page_image"]["stored_image_uri"] == result.iloc[0]["_stored_image_uri"]
        assert result.iloc[0]["page_image"]["image_b64"] == b64

    def test_store_operator_forwards_storage_options(self, monkeypatch):
        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)
        calls: list[tuple[str, str, dict]] = []

        class _Writer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, data: bytes) -> int:
                self.data = data
                return len(data)

        def _fake_open(path: str, mode: str = "rb", **kwargs):
            calls.append((path, mode, kwargs))
            return _Writer()

        monkeypatch.setattr("nemo_retriever.operators.graph_ops.store_operator.fsspec.open", _fake_open)

        params = StoreParams(
            storage_uri="s3://bucket/prefix",
            storage_options={"key": "YOUR_KEY", "secret": "YOUR_SECRET"},
        )
        result = StoreOperator(params=params).process(df)

        assert len(calls) == 1
        assert calls[0][0].startswith("s3://bucket/prefix/")
        assert calls[0][1] == "wb"
        assert calls[0][2] == {"key": "YOUR_KEY", "secret": "YOUR_SECRET"}
        assert result.iloc[0]["_stored_image_uri"].startswith("s3://bucket/prefix/")

    def test_store_object_key_uses_image_hash_without_mutating_source_columns(self, monkeypatch):
        b64 = _make_tiny_png_b64()
        raw = base64.b64decode(b64)
        df = _make_embedded_df(b64)
        df.at[0, "path"] = "/nested/source folders/report with spaces.pdf"
        df.at[0, "page_number"] = 0
        df.at[0, "_content_type"] = "text/page"
        calls: list[str] = []

        class _Writer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, data: bytes) -> int:
                return len(data)

        def _fake_open(path: str, mode: str = "rb", **kwargs):
            calls.append(path)
            return _Writer()

        monkeypatch.setattr("nemo_retriever.operators.graph_ops.store_operator.fsspec.open", _fake_open)

        result = StoreOperator(params=StoreParams(storage_uri="memory://stored")).process(df)

        assert result.iloc[0]["path"] == "/nested/source folders/report with spaces.pdf"
        assert result.iloc[0]["page_number"] == 0
        assert result.iloc[0]["_content_type"] == "text/page"
        assert calls
        assert calls[0] == (
            f"memory://stored/report_with_spaces-p0-text_page-{hashlib.sha1(raw).hexdigest()}.png"
        )

    def test_row_image_uris_are_independent_of_batch_order(self, tmp_path: Path):
        first = _make_tiny_png_b64()
        second = base64.b64encode(b"different image bytes").decode()
        rows = pd.DataFrame(
            {
                "path": ["report.pdf", "report.pdf"],
                "page_number": [1, 1],
                "_content_type": ["image", "image"],
                "_image_b64": [first, second],
            }
        )
        params = StoreParams(storage_uri=str(tmp_path))

        original = StoreOperator(params=params).process(rows)
        reversed_rows = StoreOperator(params=params).process(rows.iloc[::-1].reset_index(drop=True))

        assert original["_stored_image_uri"].tolist() == reversed_rows["_stored_image_uri"].tolist()[::-1]
        assert len(list(tmp_path.iterdir())) == 2

    def test_embedding_preserves_image_b64_for_post_embed_store(self, monkeypatch):
        from nemo_retriever.models.inference import runtime

        b64 = _make_tiny_png_b64()
        df = _make_embedded_df(b64)
        df["_embed_modality"] = "text_image"

        def _fake_embed_group(group_df: pd.DataFrame, **kwargs):
            out = group_df.copy()
            out["metadata"] = [{"embedding": [0.1, 0.2]} for _ in range(len(out.index))]
            return out

        monkeypatch.setattr(runtime, "_embed_group", _fake_embed_group)

        result = runtime.embed_text_main_text_embed(df, model=object(), embed_modality="text_image")

        assert result.iloc[0]["_image_b64"] == b64
        assert "_embed_modality" not in result.columns

    def test_store_params_reject_removed_legacy_knobs(self):
        with pytest.raises(ValueError):
            StoreParams(public_base_url="https://cdn.example.com")

    def test_explode_does_not_reload_stored_uri_for_embedding(self, monkeypatch):
        from nemo_retriever.common.modality.content_transforms import (
            explode_content_to_rows,
        )

        def _fail_load(uri):
            raise AssertionError(f"content transform attempted to reload stored image URI: {uri}")

        monkeypatch.setattr("nemo_retriever.common.io.image_store.load_image_b64_from_uri", _fail_load)

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": None, "stored_image_uri": "file:///page.png"}],
            }
        )
        result = explode_content_to_rows(df, modality="text_image")

        assert result.iloc[0]["_image_b64"] is None
        assert result.iloc[0]["_stored_image_uri"] == "file:///page.png"

    def test_collapse_does_not_reload_stored_uri_for_embedding(self, monkeypatch):
        from nemo_retriever.common.modality.content_transforms import (
            collapse_content_to_page_rows,
        )

        def _fail_load(uri):
            raise AssertionError(f"content transform attempted to reload stored image URI: {uri}")

        monkeypatch.setattr("nemo_retriever.common.io.image_store.load_image_b64_from_uri", _fail_load)

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": None, "stored_image_uri": "file:///page.png"}],
            }
        )
        result = collapse_content_to_page_rows(df, modality="text_image")

        assert result.iloc[0]["_image_b64"] is None
        assert result.iloc[0]["_stored_image_uri"] == "file:///page.png"
