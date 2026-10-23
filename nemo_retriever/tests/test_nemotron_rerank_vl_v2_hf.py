# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for NemotronRerankVLV2, VL-aware rerank_hits, content_transforms
URI propagation, and LanceDB schema stored_image_uri support.

Heavy dependencies (torch, transformers) are stubbed so no GPU or model
download is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# NemotronRerankVLV2 — properties & score()
# ---------------------------------------------------------------------------


class TestNemotronRerankVLV2Properties:
    def _make_instance(self, model_name="nvidia/llama-nemotron-rerank-vl-1b-v2"):
        from nemo_retriever.models.local import nemotron_rerank_vl_v2_hf as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.AutoProcessor") as MockProc,
            patch("transformers.AutoModelForSequenceClassification") as MockModel,
        ):
            proc = MockProc.from_pretrained.return_value
            proc.tokenizer.eos_token_id = 0
            mdl = MockModel.from_pretrained.return_value
            mdl.eval.return_value = mdl
            mdl.to.return_value = mdl
            mdl.config.pad_token_id = 1
            obj = mod.NemotronRerankVLV2(model_name=model_name)
        return obj

    def test_model_name(self):
        obj = self._make_instance()
        assert obj.model_name == "nvidia/llama-nemotron-rerank-vl-1b-v2"

    def test_model_type(self):
        obj = self._make_instance()
        assert obj.model_type == "vl_reranker"

    def test_model_runmode(self):
        obj = self._make_instance()
        assert obj.model_runmode == "local"

    def test_input_batch_size(self):
        obj = self._make_instance()
        assert obj.input_batch_size == 32


class TestNemotronRerankVLV2Score:
    @pytest.fixture()
    def reranker(self):
        from nemo_retriever.models.local import nemotron_rerank_vl_v2_hf as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value=None),
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.AutoProcessor") as MockProc,
            patch("transformers.AutoModelForSequenceClassification") as MockModel,
        ):
            proc_inst = MockProc.from_pretrained.return_value
            proc_inst.tokenizer.eos_token_id = 0
            mdl_inst = MockModel.from_pretrained.return_value
            mdl_inst.eval.return_value = mdl_inst
            mdl_inst.to.return_value = mdl_inst
            mdl_inst.config.pad_token_id = 1
            obj = mod.NemotronRerankVLV2()
        return obj

    def test_score_empty_documents_returns_empty(self, reranker):
        assert reranker.score("q", []) == []

    def test_score_pairs_empty_returns_empty(self, reranker):
        assert reranker.score_pairs([]) == []

    def test_score_calls_processor_and_model(self, reranker):
        logit_tensor = MagicMock()
        logit_tensor.view.return_value = logit_tensor
        logit_tensor.cpu.return_value = logit_tensor
        logit_tensor.tolist.return_value = [3.5, -1.2]

        model_out = MagicMock()
        model_out.logits = logit_tensor

        proc_out = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        proc_out["input_ids"].to.return_value = proc_out["input_ids"]
        proc_out["attention_mask"].to.return_value = proc_out["attention_mask"]
        reranker._processor.process_queries_documents_crossencoder.return_value = proc_out
        reranker._model.return_value = model_out

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            scores = reranker.score(
                "What is ML?",
                ["Machine learning is...", "Paris is..."],
                images_b64=["iVBOR...", None],
                batch_size=2,
            )

        assert scores == [3.5, -1.2]
        # Verify processor was called with features list
        call_args = reranker._processor.process_queries_documents_crossencoder.call_args
        features = call_args[0][0]
        assert len(features) == 2
        assert features[0]["question"] == "What is ML?"
        assert features[0]["doc_text"] == "Machine learning is..."
        assert features[0]["doc_image"] == {"base64": "iVBOR..."}
        assert features[1]["doc_image"] is None

    def test_score_without_images(self, reranker):
        logit_tensor = MagicMock()
        logit_tensor.view.return_value = logit_tensor
        logit_tensor.cpu.return_value = logit_tensor
        logit_tensor.tolist.return_value = [1.0]

        model_out = MagicMock()
        model_out.logits = logit_tensor

        proc_out = {"input_ids": MagicMock()}
        proc_out["input_ids"].to.return_value = proc_out["input_ids"]
        reranker._processor.process_queries_documents_crossencoder.return_value = proc_out
        reranker._model.return_value = model_out

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            scores = reranker.score("q", ["doc"])

        assert scores == [1.0]
        call_args = reranker._processor.process_queries_documents_crossencoder.call_args
        features = call_args[0][0]
        assert features[0]["doc_image"] is None


class TestPrepareImages:
    def test_none_images(self):
        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        result = NemotronRerankVLV2._prepare_images(None, 3)
        assert result == [None, None, None]

    def test_mixed_images(self):
        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        result = NemotronRerankVLV2._prepare_images(["abc", None, "def"], 3)
        assert result == [{"base64": "abc"}, None, {"base64": "def"}]

    def test_short_list_padded(self):
        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        result = NemotronRerankVLV2._prepare_images(["abc"], 3)
        assert len(result) == 3
        assert result[0] == {"base64": "abc"}
        assert result[1] is None
        assert result[2] is None


# ---------------------------------------------------------------------------
# is_vl_rerank_model / create_local_reranker
# ---------------------------------------------------------------------------


class TestModelRegistration:
    def test_is_vl_rerank_model_positive(self):
        from nemo_retriever.models import is_vl_rerank_model

        assert is_vl_rerank_model("nvidia/llama-nemotron-rerank-vl-1b-v2") is True
        assert is_vl_rerank_model("llama-nemotron-rerank-vl-1b-v2") is True

    def test_is_vl_rerank_model_negative(self):
        from nemo_retriever.models import is_vl_rerank_model

        assert is_vl_rerank_model("nvidia/llama-nemotron-rerank-1b-v2") is False
        assert is_vl_rerank_model(None) is False
        assert is_vl_rerank_model("") is False

    def test_create_local_reranker_dispatches_to_vl(self):
        with (patch("nemo_retriever.models.local.nemotron_rerank_vl_v2_hf.NemotronRerankVLV2") as MockVL,):
            MockVL.return_value = MagicMock()
            from nemo_retriever.models import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-vl-1b-v2", backend="hf")
            MockVL.assert_called_once()
            assert result is MockVL.return_value

    def test_create_local_reranker_dispatches_to_text(self):
        with (patch("nemo_retriever.models.local.nemotron_rerank_v2.NemotronRerankV2") as MockText,):
            MockText.return_value = MagicMock()
            from nemo_retriever.models import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-1b-v2")
            MockText.assert_called_once()
            assert result is MockText.return_value


# ---------------------------------------------------------------------------
# rerank_hits with images
# ---------------------------------------------------------------------------


class TestRerankHitsVL:
    def test_rerank_hits_loads_images_for_vl_model(self):
        from nemo_retriever.operators.rerank import rerank_hits

        model = MagicMock()
        model.model_name = "nvidia/llama-nemotron-rerank-vl-1b-v2"
        model.score.return_value = [5.0, 3.0]

        hits = [
            {"text": "doc1", "stored_image_uri": "file:///tmp/img1.png"},
            {"text": "doc2", "stored_image_uri": "file:///tmp/img2.png"},
        ]

        with patch("nemo_retriever.common.io.image_store.load_image_b64_from_uri") as mock_load:
            mock_load.side_effect = ["b64_data_1", "b64_data_2"]
            out = rerank_hits("q", hits, model=model, modality="text_image")

        assert mock_load.call_count == 2
        model.score.assert_called_once()
        call_kwargs = model.score.call_args
        assert call_kwargs.kwargs["images_b64"] == ["b64_data_1", "b64_data_2"]
        assert len(out) == 2

    def test_rerank_hits_no_images_for_text_model(self):
        from nemo_retriever.operators.rerank import rerank_hits

        model = MagicMock()
        model.model_name = "nvidia/llama-nemotron-rerank-1b-v2"
        model.score.return_value = [1.0]

        hits = [{"text": "doc1", "stored_image_uri": "file:///tmp/img.png"}]
        out = rerank_hits("q", hits, model=model)

        # Text model should NOT receive images_b64
        call_kwargs = model.score.call_args
        assert "images_b64" not in call_kwargs.kwargs
        assert len(out) == 1

    def test_rerank_hits_graceful_without_uri(self):
        from nemo_retriever.operators.rerank import rerank_hits

        model = MagicMock()
        model.model_name = "nvidia/llama-nemotron-rerank-vl-1b-v2"
        model.score.return_value = [2.0, 1.0]

        hits = [{"text": "doc1"}, {"text": "doc2", "stored_image_uri": ""}]
        out = rerank_hits("q", hits, model=model, modality="text_image")

        call_kwargs = model.score.call_args
        assert call_kwargs.kwargs["images_b64"] == [None, None]
        assert len(out) == 2


# ---------------------------------------------------------------------------
# _rerank_via_endpoint with images
# ---------------------------------------------------------------------------


class TestRerankViaEndpointVL:
    def test_endpoint_includes_images_in_payload(self):
        from nemo_retriever.operators.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rankings": [{"index": 0, "logit": 0.9}, {"index": 1, "logit": 0.3}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint(
                "q",
                ["doc1", "doc2"],
                endpoint="http://localhost:8000",
                images_b64=["b64_img", None],
            )

        payload = mock_post.call_args[1]["json"]
        assert payload["passages"][0] == {"text": "doc1", "image": "b64_img"}
        assert payload["passages"][1] == {"text": "doc2"}

    def test_endpoint_no_images_text_only_payload(self):
        from nemo_retriever.operators.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"rankings": [{"index": 0, "logit": 0.5}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint("q", ["doc"], endpoint="http://localhost:8000")

        payload = mock_post.call_args[1]["json"]
        assert payload["passages"] == [{"text": "doc"}]


# ---------------------------------------------------------------------------
# content_transforms — _stored_image_uri propagation
# ---------------------------------------------------------------------------


class TestContentTransformsStoredImageURI:
    def test_explode_propagates_page_uri_to_text_row(self):
        from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": "abc", "stored_image_uri": "file:///page.png"}],
            }
        )
        result = explode_content_to_rows(df, modality="text_image")
        assert "_stored_image_uri" in result.columns
        assert result.iloc[0]["_stored_image_uri"] == "file:///page.png"

    def test_explode_propagates_item_uri_to_structured_row(self):
        from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": "abc", "stored_image_uri": "file:///page.png"}],
                "table": [
                    [
                        {
                            "text": "Table text",
                            "stored_image_uri": "file:///table_0.png",
                        }
                    ]
                ],
            }
        )
        result = explode_content_to_rows(df, modality="text_image")
        table_rows = result[result["_content_type"] == "table"]
        assert len(table_rows) == 1
        assert table_rows.iloc[0]["_stored_image_uri"] == "file:///table_0.png"

    def test_explode_falls_back_to_page_uri_for_item_without_uri(self):
        from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": "abc", "stored_image_uri": "file:///page.png"}],
                "table": [[{"text": "No URI table"}]],
            }
        )
        result = explode_content_to_rows(df, modality="text_image")
        table_rows = result[result["_content_type"] == "table"]
        assert table_rows.iloc[0]["_stored_image_uri"] == "file:///page.png"

    def test_explode_without_store_uri_is_none(self):
        from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

        df = pd.DataFrame({"text": ["Page text"], "page_image": [{"image_b64": "abc"}]})
        result = explode_content_to_rows(df, modality="text_image")
        assert result.iloc[0]["_stored_image_uri"] is None

    def test_collapse_propagates_page_uri(self):
        from nemo_retriever.common.modality.content_transforms import (
            collapse_content_to_page_rows,
        )

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": "abc", "stored_image_uri": "file:///page.png"}],
            }
        )
        result = collapse_content_to_page_rows(df, modality="text_image")
        assert "_stored_image_uri" in result.columns
        assert result.iloc[0]["_stored_image_uri"] == "file:///page.png"

    def test_no_content_columns_propagates_uri(self):
        """When there are no content columns to explode, URI should still propagate."""
        from nemo_retriever.common.modality.content_transforms import explode_content_to_rows

        df = pd.DataFrame(
            {
                "text": ["Page text"],
                "page_image": [{"image_b64": "abc", "stored_image_uri": "file:///page.png"}],
            }
        )
        result = explode_content_to_rows(df, modality="text")
        assert "_stored_image_uri" in result.columns
        assert result.iloc[0]["_stored_image_uri"] == "file:///page.png"


# ---------------------------------------------------------------------------
# lancedb_schema — stored_image_uri in schema and row building
# ---------------------------------------------------------------------------


class TestLanceDBStoredImageURI:
    def test_schema_includes_stored_image_uri(self):
        from nemo_retriever.common.vdb.lancedb_schema import lancedb_schema

        schema = lancedb_schema()
        field_names = [f.name for f in schema]
        assert "stored_image_uri" in field_names

    def test_build_lancedb_row_includes_stored_image_uri(self):
        from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row

        row = MagicMock()
        row.embedding = [0.1] * 2048
        row.path = "/tmp/test.pdf"
        row.page_number = 1
        row.text = "hello"
        row._stored_image_uri = "file:///tmp/page_1.png"
        row._content_type = "text"
        row._bbox_xyxy_norm = None

        result = build_lancedb_row(row, embedding=row.embedding)
        assert result is not None
        assert result["stored_image_uri"] == "file:///tmp/page_1.png"
        assert result["content_type"] == "text"
        assert result["bbox_xyxy_norm"] == ""

    def test_build_lancedb_row_empty_when_no_uri(self):
        from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row

        row = MagicMock()
        row.embedding = [0.1] * 2048
        row.path = "/tmp/test.pdf"
        row.page_number = 1
        row.text = "hello"
        row._stored_image_uri = None
        row._content_type = None
        row._bbox_xyxy_norm = None

        result = build_lancedb_row(row, embedding=row.embedding)
        assert result is not None
        assert result["stored_image_uri"] == ""

    def test_build_lancedb_row_missing_attr(self):
        from nemo_retriever.common.vdb.lancedb_schema import build_lancedb_row

        row = MagicMock(spec=["embedding", "metadata", "path", "page_number", "text"])
        row.embedding = [0.1] * 2048
        row.path = "/tmp/test.pdf"
        row.page_number = 1
        row.text = "hello"

        result = build_lancedb_row(row, embedding=row.embedding)
        assert result is not None
        assert result["stored_image_uri"] == ""
