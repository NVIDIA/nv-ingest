# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare consumers across compact/rich boundaries, for built-in ingestion stages."""
from __future__ import annotations

import base64
from io import BytesIO
from itertools import product

import pandas as pd
from PIL import Image
import pytest

from nemo_retriever.common.modality.content_transforms import (
    _compact_embedding_rows,
    collapse_content_to_page_rows,
    explode_content_to_rows,
)
from nemo_retriever.common.schemas.embedding import embedding_text_input
from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy, prepare_embedding_inputs


class CharacterTokenizer:
    """Reversible tokens make whitespace and exact split reconstruction observable."""

    def encode(self, value, **kwargs):
        return list(map(ord, value))

    def decode(self, value, **kwargs):
        return "".join(map(chr, value))


def _admit(frame):
    return prepare_embedding_inputs(
        frame, policy=EmbeddingInputPolicy(CharacterTokenizer(), max_tokens=8, prefix="")
    ).frame


def _inputs(frame):
    return [embedding_text_input(row.to_dict()) for _, row in frame.iterrows()]


def _records(frame):
    frame = frame.copy(deep=True)
    frame["text_embeddings_1b_v2"] = [{"embedding": [1.0, float(i)]} for i in range(len(frame))]
    return to_client_vdb_records(frame)


MODES = [("page", m, m) for m in ("text", "image", "text_image")] + [
    ("element", t, s) for t, s in product(("text", "image", "text_image"), repeat=2)
]


@pytest.mark.parametrize("granularity,text_mode,structured_mode", MODES)
@pytest.mark.parametrize("caption", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_modalities_captions_and_image_uris_preserve_consumer_contract(
    tmp_path, granularity, text_mode, structured_mode, caption, stored
):
    buf = BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    page_image = {"image_b64": encoded}
    if stored:
        path = tmp_path / "page.png"
        path.write_bytes(buf.getvalue())
        page_image["stored_image_uri"] = str(path)
    source = {
        "text": "Page prose with overflow text",
        "page_image": page_image,
        "path": "/docs/multimodal.pdf",
        "page_number": 2,
        "metadata": {"custom": {"a": [1, 2]}, "source_path": "/docs/multimodal.pdf"},
        "page_elements_v3_counts_by_label": {"table": 1, "chart": 1, "infographic": 1},
    }
    for kind in ("table", "chart", "infographic", "images"):
        item = {"text": f"{kind} content", "bbox_xyxy_norm": [0.1, 0.1, 0.7, 0.8]}
        if caption:
            item["caption"] = f"{kind} caption"
        source[kind] = [item]
    columns = ("table", "chart", "infographic") + (("images",) if caption else ())
    if granularity == "page":
        reshape = collapse_content_to_page_rows
        kwargs = {"modality": text_mode, "content_columns": columns}
    else:
        reshape = explode_content_to_rows
        kwargs = dict(
            text_elements_modality=text_mode,
            structured_elements_modality=structured_mode,
            content_columns=columns,
        )
    rich = reshape(pd.DataFrame([source]), **kwargs)
    compact = reshape(pd.DataFrame([source]), compact=True, **kwargs)
    assert all("image_b64" not in value for value in compact["page_image"])
    assert not set(columns).intersection(compact.columns)
    assert _inputs(compact) == _inputs(rich)
    if "_image_b64" in rich:
        assert compact["_image_b64"].fillna("").tolist() == rich["_image_b64"].fillna("").tolist()
        assert any(isinstance(x, str) and x for x in compact["_image_b64"])
    assert compact["_stored_image_uri"].fillna("").tolist() == rich["_stored_image_uri"].fillna("").tolist()
    assert _records(compact) == _records(rich)
    admitted_rich, admitted_compact = _admit(rich), _admit(compact)
    assert _inputs(admitted_compact) == _inputs(admitted_rich)
    assert _records(admitted_compact) == _records(admitted_rich)


@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("strip_base64", [False, True])
def test_upstream_store_and_dedup_keep_required_rich_payload(tmp_path, dedup, strip_base64):
    from nemo_retriever.common.params import StoreParams
    from nemo_retriever.operators.dedup import dedup_images
    from nemo_retriever.operators.graph_ops.store_operator import StoreOperator

    buf = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    item = {"image_b64": encoded, "caption": "Figure caption"}
    frame = pd.DataFrame(
        [
            {
                "text": "body",
                "path": "/docs/store.pdf",
                "page_number": 1,
                "page_image": {"image_b64": encoded},
                "images": [item.copy(), item.copy()],
            }
        ]
    )
    if dedup:
        frame = dedup_images(frame, content_hash=True, bbox_iou=False)
    assert len(frame.iloc[0]["images"]) == (1 if dedup else 2)
    stored = StoreOperator(params=StoreParams(storage_uri=str(tmp_path), strip_base64=strip_base64)).process(frame)
    assert list(tmp_path.glob("*.png"))
    assert bool(stored.iloc[0]["page_image"].get("image_b64")) != strip_base64
    rich = explode_content_to_rows(stored, modality="text_image", content_columns=("images",))
    compact = _compact_embedding_rows(rich)
    assert _inputs(compact) == _inputs(rich)
    assert compact["_image_b64"].fillna("").tolist() == rich["_image_b64"].fillna("").tolist()
    assert _records(compact) == _records(rich)


@pytest.mark.parametrize("caption,dedup,store", list(product((False, True), repeat=3)))
def test_optional_consumers_precede_compact_reshape(tmp_path, caption, dedup, store):
    from nemo_retriever.common.params import CaptionParams, DedupParams, EmbedParams, StoreParams, VdbUploadParams
    from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph
    from nemo_retriever.graph.executor import RayDataExecutor

    graph = build_post_extract_graph(
        embed_params=EmbedParams(),
        caption_params=CaptionParams() if caption else None,
        dedup_params=DedupParams() if dedup else None,
        store_params=StoreParams(storage_uri=str(tmp_path)) if store else None,
        vdb_upload_params=VdbUploadParams(),
        compact_embedding_transport=True,
    )
    nodes = RayDataExecutor(graph)._linearize(graph)
    reshape = nodes[-3].operator
    assert reshape.name == "ExplodeContentToRows"
    assert reshape.fn.keywords["compact"] is True
    assert len(nodes[:-3]) == sum((caption, dedup, store))


@pytest.mark.parametrize("case", ["full_results", "after_embed", "custom_text", "custom_output", "no_sink"])
def test_other_graph_shapes_keep_rich_rows(tmp_path, case):
    from nemo_retriever.common.params import EmbedParams, StoreParams, VdbUploadParams
    from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph
    from nemo_retriever.graph.executor import RayDataExecutor

    orders = {"after_embed": ("embed", "store")}
    graph = build_post_extract_graph(
        embed_params=EmbedParams(
            text_column="custom" if case == "custom_text" else "text",
            output_column="custom" if case == "custom_output" else "text_embeddings_1b_v2",
        ),
        store_params=StoreParams(storage_uri=str(tmp_path)) if case == "after_embed" else None,
        stage_order=orders.get(case, ()),
        vdb_upload_params=VdbUploadParams() if case != "no_sink" else None,
        compact_embedding_transport=case != "full_results",
    )
    nodes = RayDataExecutor(graph)._linearize(graph)
    for node in nodes:
        if getattr(node.operator, "name", None) == "ExplodeContentToRows":
            output = node.operator.process(pd.DataFrame([{"text": "body", "page_image": {"image_b64": "data"}}]))
            assert output.iloc[0]["page_image"]["image_b64"] == "data"


@pytest.mark.parametrize("reshape", [collapse_content_to_page_rows, explode_content_to_rows])
def test_image_only_rows_keep_embedding_input_and_stored_record(reshape):
    frame = pd.DataFrame([{"text": "", "page_image": {"image_b64": "encoded"}, "path": "scan.pdf", "page_number": 3}])
    rich = reshape(frame, modality="image")
    compact = reshape(frame, modality="image", compact=True)
    assert compact["_image_b64"].tolist() == rich["_image_b64"].tolist() == ["encoded"]
    assert _records(compact) == _records(rich)


@pytest.mark.parametrize("identity", ["id", "content_metadata", "top_level_id", "element_id"])
def test_builtin_metadata_preserves_overflow_identity_and_exact_text(identity):
    exact = "alpha \n omega and further text"
    metadata = {"id": "element-A"} if identity == "id" else {"content_metadata": {"id": "element-A"}}
    frame = pd.DataFrame([{"text": exact, "metadata": metadata, "path": "doc.pdf", "page_number": 7}])
    if identity in {"top_level_id", "element_id"}:
        frame["metadata"] = [{}]
        frame["id" if identity == "top_level_id" else "element_id"] = "element-A"
    frame["custom_column"] = "retained"
    rich, compact = _admit(frame), _admit(_compact_embedding_rows(frame))
    assert compact["custom_column"].tolist() == rich["custom_column"].tolist()
    assert len(rich) > 1
    assert "".join(_inputs(compact)) == exact
    assert _inputs(compact) == _inputs(rich)
    assert compact["metadata"].tolist() == rich["metadata"].tolist()
    assert _records(compact) == _records(rich)


def test_compaction_preserves_detection_metadata_precedence_without_mutating_source():
    metadata = {"custom": "retained", "content_metadata": {"ocr_table_detections": 7}}
    frame = pd.DataFrame(
        [
            {
                "text": "body",
                "metadata": metadata,
                "table": [{"text": "table"}],
                "chart": [],
                "page_elements_v3": {"regions": [{"label": "table"}]},
                "table_structure_v1": {"image_b64": "large raster"},
                "table_structure_ocr_v1": {"regions": []},
            }
        ]
    )
    compact = _compact_embedding_rows(frame)
    assert not {"table", "chart", "page_elements_v3", "table_structure_v1", "table_structure_ocr_v1"}.intersection(
        compact.columns
    )
    assert compact.iloc[0]["metadata"] == {
        "custom": "retained",
        "content_metadata": {"ocr_table_detections": 7, "ocr_chart_detections": 0},
    }
    assert metadata == {"custom": "retained", "content_metadata": {"ocr_table_detections": 7}}
    assert _records(compact) == _records(frame)


def test_errors_keep_original_stage_payload_and_failure_details():
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor, GraphIngestionError
    from nemo_retriever.common.vdb.records import VdbUploadError

    frame = pd.DataFrame(
        [
            {
                "text": "valid text",
                "path": "doc.pdf",
                "page_elements_v3": {
                    "regions": [{"error": {"stage": "page_elements_v3", "type": "RuntimeError", "message": "failed"}}]
                },
                "page_image": {"image_b64": "discarded"},
            }
        ]
    )
    compact = _compact_embedding_rows(frame)
    assert all("image_b64" not in value for value in compact["page_image"])
    assert compact["page_elements_v3"].tolist() == frame["page_elements_v3"].tolist()
    ingestor = GraphIngestor(run_mode="batch").extract(page_elements_invoke_url="http://invalid.test")
    failures = []
    for batch in (frame, compact):
        with pytest.raises(GraphIngestionError) as failure:
            ingestor._raise_for_stage_errors(batch)
        failures.append(str(failure.value))
    assert failures[0] == failures[1]
    conversion_failures = []
    for batch in (frame, compact):
        with pytest.raises(VdbUploadError) as failure:
            to_client_vdb_records(batch)
        conversion_failures.append(str(failure.value))
    assert conversion_failures[0] == conversion_failures[1]
