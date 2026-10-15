# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying all pipeline actors inherit from AbstractOperator."""

import inspect
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from nemo_retriever.operators.abstract_operator import AbstractOperator


# ---------------------------------------------------------------------------
# 1. PDFSplitActor
# ---------------------------------------------------------------------------
class TestPDFSplitActor:
    def _make(self):
        from nemo_retriever.operators.extract.pdf.split import PDFSplitActor

        return PDFSplitActor()

    def test_inherits(self):
        from nemo_retriever.operators.extract.pdf.split import PDFSplitActor

        assert issubclass(PDFSplitActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.operators.extract.pdf.split.split_pdf_batch")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"page": [1]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.pdf.split.split_pdf_batch")
    def test_call_delegates_to_run(self, mock_fn):
        expected = pd.DataFrame({"page": [1]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"x"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 2. PDFExtractionActor
# ---------------------------------------------------------------------------
class TestPDFExtractionActor:
    def _make(self):
        from nemo_retriever.operators.extract.pdf.extract import PDFExtractionActor

        return PDFExtractionActor(method="pdfium")

    def test_inherits(self):
        from nemo_retriever.operators.extract.pdf.extract import PDFExtractionActor

        assert issubclass(PDFExtractionActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.operators.extract.pdf.extract.pdf_extraction")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"bytes": [b"x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.pdf.extract.pdf_extraction")
    def test_call_delegates_to_run(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"x"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.pdf.extract.pdf_extraction", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"x"], "path": ["/tmp/a.pdf"]})
        result = actor(df)
        assert isinstance(result, list)
        record = result[0]
        assert record["metadata"]["error"]["type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# 3. PageElementDetectionActor
# ---------------------------------------------------------------------------
class TestPageElementDetectionActor:
    def _make(self):
        from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor

        return PageElementDetectionActor(invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor

        assert issubclass(PageElementDetectionActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.operators.extract.page_elements.cpu_actor.detect_page_elements_v3")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.page_elements.cpu_actor.detect_page_elements_v3")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"page_elements_v3": ["det"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"page_image": ["x"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch(
        "nemo_retriever.operators.extract.page_elements.cpu_actor.detect_page_elements_v3",
        side_effect=RuntimeError("boom"),
    )
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "page_elements_v3" in result.columns


# ---------------------------------------------------------------------------
# 5. TableStructureActor
# ---------------------------------------------------------------------------
class TestTableStructureActor:
    def _make(self):
        from nemo_retriever.operators.extract.table.table_detection import TableStructureActor

        return TableStructureActor(table_structure_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.operators.extract.table.table_detection import TableStructureActor

        assert issubclass(TableStructureActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.operators.extract.table.cpu_actor.table_structure_ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"table": [[]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch(
        "nemo_retriever.operators.extract.table.cpu_actor.table_structure_ocr_page_elements",
        side_effect=RuntimeError("boom"),
    )
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "table_structure_ocr_v1" in result.columns


# ---------------------------------------------------------------------------
# 5b. TableStructureActor (GPU variant) regression tests
# ---------------------------------------------------------------------------
class TestTableStructureGPUActor:
    """Regression tests for the GPU variant of TableStructureActor.

    The GPU variant lives in nemo_retriever.operators.extract.table.gpu_actor and is selected
    by the archetype resolver when GPUs are available and no CPU-only
    endpoint is configured. Prior to this fix, its __init__ referenced
    ``self._ocr_invoke_url`` without ever assigning it, raising
    ``AttributeError`` for any non-CPU dispatch. These tests pin the
    contract so the regression cannot reappear.
    """

    def test_init_signature_uses_ocr_version_selector(self):
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        assert set(inspect.signature(GPUActor.__init__).parameters) == {
            "self",
            "table_structure_invoke_url",
            "ocr_invoke_url",
            "ocr_version",
            "ocr_lang",
            "invoke_url",
            "api_key",
            "table_output_format",
            "request_timeout_s",
            "remote_max_pool_workers",
            "remote_max_retries",
            "remote_max_429_retries",
        }

    def test_init_with_no_kwargs_defaults_to_local_ocr_v2(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor()

        assert actor._table_structure_invoke_url == ""
        assert actor._ocr_invoke_url == ""
        mock_ts.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="multi")
        assert actor._nim_client is None

    def test_init_can_explicitly_use_local_ocr_v1(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)

        actor = GPUActor(ocr_version="v1")

        assert actor._ocr_invoke_url == ""
        mock_ts.assert_called_once_with()
        mock_ocr_v2.assert_called_once_with(lang="v1")
        assert actor._nim_client is None

    def test_init_with_ocr_invoke_url_skips_local_ocr(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        mock_ts = MagicMock()
        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronTableStructureV1", mock_ts)
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(ocr_invoke_url="http://ocr.example/v1/cv/nvidia/nemotron-ocr-v2")

        assert actor._ocr_invoke_url == "http://ocr.example/v1/cv/nvidia/nemotron-ocr-v2"
        assert actor._ocr_model is None
        mock_ocr.assert_not_called()
        mock_ts.assert_called_once_with()
        assert actor._nim_client is not None

    def test_init_with_both_urls_skips_all_local_models(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url="http://ocr.example/v1",
        )

        assert actor._table_structure_model is None
        assert actor._ocr_model is None
        mock_ocr.assert_not_called()
        assert actor._nim_client is not None

    def test_init_strips_whitespace_from_ocr_invoke_url(self):
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url="  http://ocr.example/v1  ",
        )

        assert actor._ocr_invoke_url == "http://ocr.example/v1"

    def test_init_treats_none_ocr_invoke_url_as_empty(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.table.gpu_actor import TableStructureActor as GPUActor

        mock_ocr = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr)

        actor = GPUActor(
            table_structure_invoke_url="http://ts.example/v1",
            ocr_invoke_url=None,
        )

        assert actor._ocr_invoke_url == ""
        mock_ocr.assert_called_once_with(lang="multi")


# ---------------------------------------------------------------------------
# 6. OCRActor
# ---------------------------------------------------------------------------
class TestOCRActor:
    def _make(self):
        from nemo_retriever.operators.extract.ocr.ocr import OCRActor

        return OCRActor(ocr_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.operators.extract.ocr.ocr import OCRActor

        assert issubclass(OCRActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.operators.extract.ocr.cpu_ocr.ocr_page_elements")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"ocr": ["res"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.ocr.cpu_ocr.ocr_page_elements", side_effect=RuntimeError("boom"))
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "ocr" in result.columns

    def test_local_gpu_actor_defaults_to_v2_multi(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor()

        mock_ocr_v2.assert_called_once_with(lang="multi")
        assert actor._nim_client is None

    def test_local_gpu_actor_passes_v2_ocr_lang(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor(ocr_lang="english")

        mock_ocr_v2.assert_called_once_with(lang="english")
        assert actor._nim_client is None

    def test_local_gpu_actor_uses_v2_legacy_mode_for_v1(self, monkeypatch):
        import nemo_retriever.models.local as local_models
        from nemo_retriever.operators.extract.ocr.gpu_ocr import OCRActor as OCRGPUActor

        mock_ocr_v2 = MagicMock()
        monkeypatch.setitem(local_models.__dict__, "NemotronOCRV2", mock_ocr_v2)
        actor = OCRGPUActor(ocr_version="v1")

        mock_ocr_v2.assert_called_once_with(lang="v1")
        assert actor._nim_client is None


# ---------------------------------------------------------------------------
# 7. NemotronParseActor
# ---------------------------------------------------------------------------
class TestNemotronParseActor:
    def _make(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor

        return NemotronParseActor(nemotron_parse_invoke_url="http://fake")

    def test_inherits(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor

        assert issubclass(NemotronParseActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    @patch("nemo_retriever.operators.extract.parse.nemotron_parse.nemotron_parse_pages")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"nemotron_parse_v1_2": ["res"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"page_image": ["x"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    def test_remote_chat_completions_uses_v1_2_protocol(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import (
            NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
            NEMOTRON_PARSE_REMOTE_DEFAULT_MODEL,
            nemotron_parse_pages,
        )

        class _FakeNIMClient:
            def __init__(self):
                self.kwargs = None

            def invoke_chat_completions_images(self, **kwargs):
                self.kwargs = kwargs
                return ["<x_0><y_0>Hello world<x_1><y_1><class_Text>"]

        client = _FakeNIMClient()
        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})

        result = nemotron_parse_pages(
            df,
            invoke_url="http://nemotron-parse:8000/v1/chat/completions",
            extract_text=True,
            nim_client=client,
        )

        assert result["text"].tolist() == ["Hello world"]
        assert client.kwargs["model"] == NEMOTRON_PARSE_REMOTE_DEFAULT_MODEL
        assert client.kwargs["task_prompt"] == NEMOTRON_PARSE_DEFAULT_TASK_PROMPT
        assert client.kwargs["extra_body"] == {"max_tokens": 8192}
        assert client.kwargs["repetition_penalty"] == 1.1

    def test_remote_chat_completions_uses_v2_protocol(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import (
            NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
            NEMOTRON_PARSE_V2_MODEL,
            nemotron_parse_pages,
        )

        class _FakeNIMClient:
            def __init__(self):
                self.kwargs = None

            def invoke_chat_completions_images(self, **kwargs):
                self.kwargs = kwargs
                return ["<x_0><y_0>Chart text<x_1><y_1><class_Chart>"]

        client = _FakeNIMClient()
        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})

        result = nemotron_parse_pages(
            df,
            invoke_url="http://nemotron-parse:8000/v1/chat/completions",
            nemotron_parse_model=NEMOTRON_PARSE_V2_MODEL,
            extract_charts=True,
            nim_client=client,
        )

        assert len(result.at[0, "chart"]) == 1
        assert client.kwargs["model"] == NEMOTRON_PARSE_V2_MODEL
        assert client.kwargs["task_prompt"] == NEMOTRON_PARSE_DEFAULT_TASK_PROMPT
        assert client.kwargs["extra_body"] == {
            "max_tokens": 8192,
            "top_k": 1,
            "skip_special_tokens": False,
        }
        assert client.kwargs["repetition_penalty"] == 1.1

    def test_remote_chat_completions_uses_tagged_protocol_for_unknown_version(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

        class _FakeNIMClient:
            def __init__(self):
                self.kwargs = None

            def invoke_chat_completions_images(self, **kwargs):
                self.kwargs = kwargs
                return ["<x_0><y_0>Future text<x_1><y_1><class_Text>"]

        client = _FakeNIMClient()
        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})

        result = nemotron_parse_pages(
            df,
            invoke_url="http://nemotron-parse:8000/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse-v1.10",
            extract_text=True,
            nim_client=client,
        )

        assert result["text"].tolist() == ["Future text"]
        assert client.kwargs["task_prompt"] is not None
        assert client.kwargs["extra_body"] == {"max_tokens": 8192}

    def test_hosted_build_contract_is_image_only_and_routes_nested_tool_json(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

        class _FakeNIMClient:
            def __init__(self):
                self.kwargs = None

            def invoke_chat_completions_images(self, **kwargs):
                self.kwargs = kwargs
                elements = [
                    {"type": "Text", "bbox": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}, "text": "Hosted text"},
                    {"type": "Table", "bbox": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2}, "text": "A | B"},
                    {"type": "Chart", "bbox": {"xmin": 2, "ymin": 2, "xmax": 3, "ymax": 3}, "text": "Chart text"},
                    {"type": "Picture", "bbox": {"xmin": 3, "ymin": 3, "xmax": 4, "ymax": 4}, "text": "Picture text"},
                ]
                return [json.dumps({"result": [elements]})]

        client = _FakeNIMClient()
        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})
        result = nemotron_parse_pages(
            df,
            invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=True,
            nim_client=client,
        )

        assert client.kwargs["model"] == "nvidia/nemotron-parse"
        assert client.kwargs["task_prompt"] is None
        assert client.kwargs["repetition_penalty"] is None
        assert client.kwargs["extra_body"] == {"max_tokens": 8192}
        assert result["text"].tolist() == ["Hosted text"]
        assert len(result.at[0, "table"]) == 1
        assert len(result.at[0, "chart"]) == 1
        assert len(result.at[0, "infographic"]) == 1

    @pytest.mark.parametrize(
        ("endpoint", "model", "expected_model", "expected_profile"),
        [
            ("https://integrate.api.nvidia.com/v1/chat/completions", None, "nvidia/nemotron-parse", "hosted_tool_call"),
            ("http://parse:8000/v1/chat/completions", None, "nvidia/nemotron-parse-v1.2", "v1_2_tagged"),
            (
                "https://integrate.api.nvidia.com/v1/chat/completions",
                "nvidia/nemotron-parse-2.0",
                "nvidia/nemotron-parse-2.0",
                "hosted_v2_0_tagged",
            ),
            (
                "http://parse:8000/v1/chat/completions",
                "nvidia/nemotron-parse",
                "nvidia/nemotron-parse",
                "hosted_tool_call",
            ),
            (
                "http://parse:8000/v1/chat/completions",
                "nvidia/nemotron-parse-v2.0",
                "nvidia/nemotron-parse-v2.0",
                "v2_0_tagged",
            ),
            (
                "http://parse:8000/v1/chat/completions",
                "nvidia/nemotron-parse-v1.2",
                "nvidia/nemotron-parse-v1.2",
                "v1_2_tagged",
            ),
            (
                "http://parse:8000/v1/chat/completions",
                "nvidia/nemotron-parse-v1.10",
                "nvidia/nemotron-parse-v1.10",
                "v1_2_tagged",
            ),
            ("http://parse:8000/v1/chat/completions", "custom/parse", "custom/parse", "v1_2_tagged"),
        ],
    )
    def test_contract_resolution(self, endpoint, model, expected_model, expected_profile):
        from nemo_retriever.operators.extract.parse.nemotron_parse import _resolve_nemotron_parse_contract

        contract = _resolve_nemotron_parse_contract(endpoint, model)

        assert contract.model == expected_model
        assert contract.profile.value == expected_profile

    @pytest.mark.parametrize(
        "model",
        [None, "nvidia/nemotron-parse", "nvidia/nemotron-parse-v1.2"],
    )
    def test_contract_resolution_rejects_mixed_endpoints(self, model):
        from nemo_retriever.operators.extract.parse.nemotron_parse import _resolve_nemotron_parse_contract

        endpoints = "https://integrate.api.nvidia.com/v1/chat/completions," "http://parse:8000/v1/chat/completions"
        with pytest.raises(ValueError, match="cannot mix NVIDIA Build and self-hosted"):
            _resolve_nemotron_parse_contract(endpoints, model)

    def test_remote_actors_reject_mixed_endpoints_before_client_creation(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import (
            NemotronParseCPUActor,
            NemotronParseGPUActor,
        )

        endpoints = "https://integrate.api.nvidia.com/v1/chat/completions," "http://parse:8000/v1/chat/completions"
        with patch("nemo_retriever.operators.extract.parse.nemotron_parse.NIMClient") as client:
            for actor_class in (NemotronParseCPUActor, NemotronParseGPUActor):
                with pytest.raises(ValueError, match="Configure a homogeneous endpoint list"):
                    actor_class(
                        nemotron_parse_invoke_url=endpoints,
                        nemotron_parse_model="nvidia/nemotron-parse-v1.2",
                    )

        client.assert_not_called()

    def test_pages_reject_mixed_endpoints_without_input_rows(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

        endpoints = "https://integrate.api.nvidia.com/v1/chat/completions," "http://parse:8000/v1/chat/completions"
        with pytest.raises(ValueError, match="One `nemotron_parse_model` and request contract"):
            nemotron_parse_pages(
                pd.DataFrame(),
                invoke_url=endpoints,
                nemotron_parse_model="nvidia/nemotron-parse-v1.2",
            )

    def test_forced_v1_2_build_text_rejection_reports_contract_mismatch(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

        class _RejectingNIMClient:
            def invoke_chat_completions_images(self, **kwargs):
                raise RuntimeError("Content cannot be a plain string; model does not support text input")

        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})
        result = nemotron_parse_pages(
            df,
            invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse-v1.2",
            nim_client=_RejectingNIMClient(),
        )

        error = result.at[0, "nemotron_parse_v1_2"]["error"]
        assert error["type"] == "ValueError"
        assert "model/contract mismatch" in error["message"]
        assert "nvidia/nemotron-parse" in error["message"]
        assert "RuntimeError: Content cannot be a plain string" in error["traceback"]
        assert "ValueError: Nemotron Parse model/contract mismatch" in error["traceback"]

    def test_forced_v2_build_text_rejection_reports_contract_mismatch(self):
        from nemo_retriever.operators.extract.parse.nemotron_parse import nemotron_parse_pages

        class _RejectingNIMClient:
            def invoke_chat_completions_images(self, **kwargs):
                raise RuntimeError("Content cannot be a plain string; model does not support text input")

        df = pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}]})
        result = nemotron_parse_pages(
            df,
            invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse-v2.0",
            nim_client=_RejectingNIMClient(),
        )

        error = result.at[0, "nemotron_parse_v1_2"]["error"]
        assert error["type"] == "ValueError"
        assert "model/contract mismatch" in error["message"]
        assert "nvidia/nemotron-parse-v2.0" in error["message"]
        assert "tagged text-control-token contract" in error["message"]

    def test_image_wrapper_can_omit_repetition_penalty(self):
        from nemo_retriever.models.nim.nim import NIMClient

        client = NIMClient(max_pool_workers=1)
        try:
            with patch.object(client, "invoke_chat_completions", return_value=["ok"]) as invoke:
                result = client.invoke_chat_completions_images(
                    invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
                    image_b64_list=["aW1hZ2U="],
                    model="nvidia/nemotron-parse",
                    repetition_penalty=None,
                    extra_body={"max_tokens": 8192},
                )
            assert result == ["ok"]
            payload_args = invoke.call_args.kwargs
            assert payload_args["messages_list"][0][0]["content"] == [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}
            ]
            assert payload_args["extra_body"] == {"max_tokens": 8192}
        finally:
            client.shutdown()

    @patch(
        "nemo_retriever.operators.extract.parse.nemotron_parse.nemotron_parse_pages", side_effect=RuntimeError("boom")
    )
    def test_call_error_handling(self, mock_fn):
        actor = self._make()
        df = pd.DataFrame({"page_image": ["x"]})
        result = actor(df)
        assert isinstance(result, pd.DataFrame)
        assert "nemotron_parse_v1_2" in result.columns


# ---------------------------------------------------------------------------
# 8. TextChunkActor
# ---------------------------------------------------------------------------
class TestTextChunkActor:
    def _make(self):
        from nemo_retriever.operators.extract.txt.ray_data import TextChunkActor

        return TextChunkActor()

    def test_inherits(self):
        from nemo_retriever.operators.extract.txt.ray_data import TextChunkActor

        assert issubclass(TextChunkActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    def test_process_empty_df(self):
        actor = self._make()
        df = pd.DataFrame()
        result = actor.process(df)
        assert result.empty

    @patch("nemo_retriever.common.modality.txt.split.split_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk1"]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"text": ["hello world"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 9. ImageLoadActor
# ---------------------------------------------------------------------------
class TestImageLoadActor:
    def _make(self):
        from nemo_retriever.operators.extract.image.ray_data import ImageLoadActor

        return ImageLoadActor()

    def test_inherits(self):
        from nemo_retriever.operators.extract.image.ray_data import ImageLoadActor

        assert issubclass(ImageLoadActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert result.empty

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"path": ["/tmp/a.png"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.operators.extract.image.ray_data.image_bytes_to_pages_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"path": ["/tmp/a.png"], "page_number": [0]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"img"], "path": ["/tmp/a.png"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.image.ray_data.image_bytes_to_pages_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"path": ["/tmp/a.png"], "page_number": [0]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"img"], "path": ["/tmp/a.png"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 10. TxtSplitActor
# ---------------------------------------------------------------------------
class TestTxtSplitActor:
    def _make(self):
        from nemo_retriever.operators.extract.txt.ray_data import TxtSplitActor

        return TxtSplitActor()

    def test_inherits(self):
        from nemo_retriever.operators.extract.txt.ray_data import TxtSplitActor

        assert issubclass(TxtSplitActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert list(result.columns) == ["text", "content", "path", "page_number", "metadata"]

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.operators.extract.txt.ray_data.txt_bytes_to_chunks_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.txt"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"hello"], "path": ["/a.txt"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.txt.ray_data.txt_bytes_to_chunks_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.txt"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"hello"], "path": ["/a.txt"]}))
        pd.testing.assert_frame_equal(result, expected)

    @patch(
        "nemo_retriever.operators.extract.txt.ray_data.text_to_chunks_df",
        side_effect=RuntimeError("tokenizer failed"),
    )
    def test_inline_failure_is_logged_with_source(self, mock_fn, caplog):
        actor = self._make()

        with caplog.at_level("WARNING", logger="nemo_retriever.operators.extract.txt.ray_data"):
            result = actor.process(pd.DataFrame({"text": ["hello"], "path": ["inline://00000000"]}))

        assert result.empty
        record = next(record for record in caplog.records if "inline://00000000" in record.getMessage())
        assert record.exc_info is not None


# ---------------------------------------------------------------------------
# 11. HtmlSplitActor
# ---------------------------------------------------------------------------
class TestHtmlSplitActor:
    def _make(self):
        from nemo_retriever.operators.extract.html.ray_data import HtmlSplitActor

        return HtmlSplitActor()

    def test_inherits(self):
        from nemo_retriever.operators.extract.html.ray_data import HtmlSplitActor

        assert issubclass(HtmlSplitActor, AbstractOperator)

    def test_preprocess_empty(self):
        actor = self._make()
        result = actor.preprocess(pd.DataFrame())
        assert list(result.columns) == ["text", "path", "page_number", "metadata"]

    @patch("nemo_retriever.operators.extract.html.ray_data.html_bytes_to_chunks_df")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.html"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        df = pd.DataFrame({"bytes": [b"<p>hi</p>"], "path": ["/a.html"]})
        result = actor.process(df)
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.extract.html.ray_data.html_bytes_to_chunks_df")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["chunk"], "path": ["/a.html"], "page_number": [0], "metadata": [{}]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"bytes": [b"<p>hi</p>"], "path": ["/a.html"]}))
        pd.testing.assert_frame_equal(result, expected)


# ---------------------------------------------------------------------------
# 12. _BatchEmbedActor
# ---------------------------------------------------------------------------
class TestBatchEmbedActor:
    @pytest.fixture(autouse=True)
    def _stub_input_policy(self, monkeypatch):
        monkeypatch.setattr(
            "nemo_retriever.operators.embed.cpu_operator.ensure_embedding_input_policy_for_batch",
            lambda kwargs, frame: None,
        )

    def _make(self):
        from nemo_retriever.common.params import EmbedParams
        from nemo_retriever.operators.embed.operators import _BatchEmbedActor

        params = EmbedParams(model_name="test-model", embed_invoke_url="http://fake")
        return _BatchEmbedActor(params=params)

    def test_inherits(self):
        from nemo_retriever.operators.embed.operators import _BatchEmbedActor

        assert issubclass(_BatchEmbedActor, AbstractOperator)

    def test_preprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.preprocess(df), df)

    def test_postprocess_passthrough(self):
        actor = self._make()
        df = pd.DataFrame({"text": ["hello"]})
        pd.testing.assert_frame_equal(actor.postprocess(df), df)

    @patch("nemo_retriever.operators.embed.cpu_operator.embed_text_main_text_embed")
    def test_process(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor.process(pd.DataFrame({"text": ["hello"]}))
        mock_fn.assert_called_once()
        pd.testing.assert_frame_equal(result, expected)

    @patch("nemo_retriever.operators.embed.cpu_operator.embed_text_main_text_embed")
    def test_call_delegates(self, mock_fn):
        expected = pd.DataFrame({"text": ["hello"], "embedding": [[0.1, 0.2]]})
        mock_fn.return_value = expected
        actor = self._make()
        result = actor(pd.DataFrame({"text": ["hello"]}))
        pd.testing.assert_frame_equal(result, expected)
