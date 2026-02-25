# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused runmode.

This mode keeps GPU model stages in one Ray actor to avoid inter-stage Ray Data
handoffs. It intentionally does not support remote NIM endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import ray.data as rd

from retriever.ocr import ocr_page_elements
from retriever.page_elements import detect_page_elements_v3

from ..pdf.extract import PDFExtractionActor
from ..pdf.split import PDFSplitActor
from ..params import EmbedParams
from ..params import ExtractParams
from ..params import PdfSplitParams
from .batch import BatchIngestor
from .inprocess import embed_text_main_text_embed, explode_content_to_rows


def _assert_no_remote_endpoints(kwargs: Dict[str, Any], *, context: str) -> None:
    remote_keys = (
        "invoke_url",
        "page_elements_invoke_url",
        "ocr_invoke_url",
        "embedding_endpoint",
        "embed_invoke_url",
    )
    present = [k for k in remote_keys if str(kwargs.get(k) or "").strip()]
    if present:
        raise ValueError(
            f"fused mode does not support remote NIM endpoints ({context}); "
            f"remove these args: {', '.join(sorted(present))}"
        )


class _FusedModelActor:
    """Single actor that runs detect -> OCR -> explode -> embed locally."""

    def __init__(self, **kwargs: Any) -> None:
        _assert_no_remote_endpoints(dict(kwargs), context="actor init")

        from retriever.model.local import NemotronOCRV1, NemotronPageElementsV3
        from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import (
            LlamaNemotronEmbed1BV2Embedder,
        )

        self._detect_kwargs = {
            "inference_batch_size": int(kwargs.get("inference_batch_size", 8)),
            "output_column": str(kwargs.get("output_column", "page_elements_v3")),
            "num_detections_column": str(kwargs.get("num_detections_column", "page_elements_v3_num_detections")),
            "counts_by_label_column": str(kwargs.get("counts_by_label_column", "page_elements_v3_counts_by_label")),
        }
        self._extract_tables = bool(kwargs.get("extract_tables", False))
        self._extract_charts = bool(kwargs.get("extract_charts", False))
        self._extract_infographics = bool(kwargs.get("extract_infographics", False))
        self._embed_kwargs = {
            "model_name": kwargs.get("model_name"),
            "text_column": str(kwargs.get("text_column", "text")),
            "inference_batch_size": int(
                kwargs.get("embed_inference_batch_size", kwargs.get("inference_batch_size", 16))
            ),
            "output_column": str(kwargs.get("embed_output_column", "text_embeddings_1b_v2")),
            "embedding_dim_column": str(kwargs.get("embedding_dim_column", "text_embeddings_1b_v2_dim")),
            "has_embedding_column": str(kwargs.get("has_embedding_column", "text_embeddings_1b_v2_has_embedding")),
        }

        device = kwargs.get("device")
        hf_cache_dir = kwargs.get("hf_cache_dir")
        normalize = bool(kwargs.get("normalize", True))
        max_length = int(kwargs.get("max_length", 8192))
        model_name_raw = kwargs.get("model_name")
        model_id = model_name_raw if (isinstance(model_name_raw, str) and "/" in model_name_raw) else None

        self._page_elements_model = NemotronPageElementsV3()
        self._ocr_model = NemotronOCRV1()
        self._embed_model = LlamaNemotronEmbed1BV2Embedder(
            device=str(device) if device else None,
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
            normalize=normalize,
            max_length=max_length,
            model_id=model_id,
        )

    def __call__(self, batch_df: Any) -> Any:
        detected = detect_page_elements_v3(
            batch_df,
            model=self._page_elements_model,
            **self._detect_kwargs,
        )
        ocred = ocr_page_elements(
            detected,
            model=self._ocr_model,
            extract_tables=self._extract_tables,
            extract_charts=self._extract_charts,
            extract_infographics=self._extract_infographics,
        )
        exploded = explode_content_to_rows(ocred, text_column=str(self._embed_kwargs["text_column"]))
        embedded = embed_text_main_text_embed(exploded, model=self._embed_model, **self._embed_kwargs)
        return embedded


class FusedIngestor(BatchIngestor):
    RUN_MODE = "fused"
    _fused_extract_flags: Dict[str, Any] = {}

    def extract(self, params: ExtractParams) -> "FusedIngestor":
        """
        Configure extraction for fused execution.

        In fused mode, PDF split/extract are separate CPU stages, but all model
        stages (page-elements, OCR, embedding) run in one actor in `.embed()`.
        """
        kwargs = {
            **params.model_dump(mode="python", exclude={"remote_retry", "batch_tuning"}, exclude_none=True),
            **params.remote_retry.model_dump(mode="python", exclude_none=True),
            **params.batch_tuning.model_dump(mode="python", exclude_none=True),
        }
        _assert_no_remote_endpoints(dict(kwargs), context="extract")

        pdf_split_batch_size = kwargs.pop("pdf_split_batch_size", 1)
        pdf_extract_batch_size = kwargs.pop("pdf_extract_batch_size", 4)
        pdf_extract_num_cpus = float(kwargs.pop("pdf_extract_num_cpus", 2))
        pdf_extract_workers = int(kwargs.pop("pdf_extract_workers", max(1, self._num_cpus // 2)))

        kwargs.setdefault("extract_page_as_image", True)
        kwargs.setdefault("dpi", 200)

        self._tasks.append(("extract", dict(kwargs)))
        self._fused_extract_flags = {
            "extract_tables": bool(kwargs.get("extract_tables", False)),
            "extract_charts": bool(kwargs.get("extract_charts", False)),
            "extract_infographics": bool(kwargs.get("extract_infographics", False)),
            "inference_batch_size": int(kwargs.get("inference_batch_size", 8)),
        }

        pdf_split_actor = PDFSplitActor(
            split_params=PdfSplitParams(
                start_page=kwargs.get("start_page"),
                end_page=kwargs.get("end_page"),
            )
        )
        self._rd_dataset = self._rd_dataset.map_batches(
            pdf_split_actor,
            batch_size=pdf_split_batch_size,
            num_cpus=1,
            num_gpus=0,
            batch_format="pandas",
        )

        extraction_actor = PDFExtractionActor(**kwargs)
        self._rd_dataset = self._rd_dataset.map_batches(
            extraction_actor,
            batch_size=pdf_extract_batch_size,
            batch_format="pandas",
            num_cpus=pdf_extract_num_cpus,
            num_gpus=0,
            compute=rd.TaskPoolStrategy(size=pdf_extract_workers),
        )
        self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=64)
        return self

    def embed(self, params: EmbedParams) -> "FusedIngestor":
        """
        Run page-elements + OCR + explode + embed in one GPU actor stage.

        `fused` mode intentionally does not support remote NIM invocation.
        """
        kwargs = {
            **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True),
            **params.runtime.model_dump(mode="python", exclude_none=True),
            **params.batch_tuning.model_dump(mode="python", exclude_none=True),
            **params.fused_tuning.model_dump(mode="python", exclude_none=True),
        }
        _assert_no_remote_endpoints(dict(kwargs), context="embed")

        fused_workers = int(kwargs.pop("fused_workers", kwargs.pop("embed_workers", 1)))
        fused_batch_size = int(kwargs.pop("fused_batch_size", kwargs.pop("embed_batch_size", 64)))
        fused_cpus_per_actor = float(kwargs.pop("fused_cpus_per_actor", kwargs.pop("embed_cpus_per_actor", 1)))
        fused_gpus_per_actor = float(kwargs.pop("fused_gpus_per_actor", kwargs.pop("gpu_fused", 1.0)))

        stage_kwargs = dict(getattr(self, "_fused_extract_flags", {}))
        stage_kwargs.update(dict(kwargs))

        self._tasks.append(("embed", dict(stage_kwargs)))
        self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=max(16, fused_batch_size))
        self._rd_dataset = self._rd_dataset.map_batches(
            _FusedModelActor,
            batch_size=fused_batch_size,
            batch_format="pandas",
            num_cpus=fused_cpus_per_actor,
            num_gpus=fused_gpus_per_actor,
            compute=rd.ActorPoolStrategy(size=fused_workers),
            fn_constructor_kwargs=stage_kwargs,
        )

        logging.info(
            "Fused embed resources: workers=%d batch_size=%d cpus/actor=%.2f gpus/actor=%.2f",
            fused_workers,
            fused_batch_size,
            fused_cpus_per_actor,
            fused_gpus_per_actor,
        )
        return self
