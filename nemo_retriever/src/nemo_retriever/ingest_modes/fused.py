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

from nemo_retriever.chart.chart_detection import (
    graphic_elements_ocr_page_elements,
)
from nemo_retriever.ocr import ocr_page_elements
from nemo_retriever.page_elements import detect_page_elements_v3
from nemo_retriever.table.table_detection import table_structure_ocr_page_elements

from ..pdf.extract import PDFExtractionActor
from ..pdf.split import PDFSplitActor
from ..params import EmbedParams
from ..params import ExtractParams
from ..params import PdfSplitParams
from .batch import _BatchEmbedActor
from .batch import BatchIngestor
from .inprocess import collapse_content_to_page_rows
from .inprocess import embed_text_main_text_embed
from .inprocess import explode_content_to_rows


def _assert_no_remote_endpoints(kwargs: Dict[str, Any], *, context: str) -> None:
    remote_keys = (
        "invoke_url",
        "page_elements_invoke_url",
        "ocr_invoke_url",
        "table_structure_invoke_url",
        "graphic_elements_invoke_url",
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

        from nemo_retriever.model import create_local_embedder
        from nemo_retriever.model.local import NemotronOCRV1, NemotronPageElementsV3

        self._detect_kwargs = {
            "inference_batch_size": int(kwargs.get("inference_batch_size", 8)),
            "output_column": str(kwargs.get("output_column", "page_elements_v3")),
            "num_detections_column": str(kwargs.get("num_detections_column", "page_elements_v3_num_detections")),
            "counts_by_label_column": str(kwargs.get("counts_by_label_column", "page_elements_v3_counts_by_label")),
        }
        from nemo_retriever.application.pipeline.build_plan import validate_table_structure_flags

        self._extract_tables = bool(kwargs.get("extract_tables", False))
        self._use_table_structure = bool(kwargs.get("use_table_structure", False))
        validate_table_structure_flags(
            self._use_table_structure,
            str(kwargs.get("table_output_format", "pseudo_markdown")),
        )
        self._extract_charts = bool(kwargs.get("extract_charts", False))
        self._extract_infographics = bool(kwargs.get("extract_infographics", False))
        self._use_graphic_elements = bool(kwargs.get("use_graphic_elements", False))
        self._embed_granularity = str(kwargs.get("embed_granularity", "element"))
        self._embed_modality = str(kwargs.get("embed_modality", "text"))
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

        self._page_elements_model = NemotronPageElementsV3()
        self._ge_model = None
        if self._use_graphic_elements and self._extract_charts:
            from nemo_retriever.model.local import NemotronGraphicElementsV1

            self._ge_model = NemotronGraphicElementsV1()
        self._ocr_model = NemotronOCRV1()
        self._table_structure_model = None
        if self._extract_tables and self._use_table_structure:
            from nemo_retriever.model.local import NemotronTableStructureV1

            self._table_structure_model = NemotronTableStructureV1()
        self._embed_model = create_local_embedder(
            kwargs.get("model_name"),
            device=str(kwargs["device"]) if kwargs.get("device") else None,
            hf_cache_dir=str(kwargs["hf_cache_dir"]) if kwargs.get("hf_cache_dir") else None,
            normalize=bool(kwargs.get("normalize", True)),
            max_length=int(kwargs.get("max_length", 8192)),
        )

    def __call__(self, batch_df: Any) -> Any:
        detected = detect_page_elements_v3(
            batch_df,
            model=self._page_elements_model,
            **self._detect_kwargs,
        )

        # Charts: combined graphic-elements + OCR stage (like table-structure + OCR).
        ocr_extract_charts = self._extract_charts
        if self._use_graphic_elements and self._extract_charts:
            detected = graphic_elements_ocr_page_elements(
                detected,
                graphic_elements_model=self._ge_model,
                ocr_model=self._ocr_model,
            )
            ocr_extract_charts = False  # Charts already handled.

        if self._extract_tables and self._use_table_structure:
            # Tables go through combined table-structure + OCR stage.
            detected = table_structure_ocr_page_elements(
                detected,
                table_structure_model=self._table_structure_model,
                ocr_model=self._ocr_model,
            )
            # Charts/infographics still go through OCR-only.
            ocred = ocr_page_elements(
                detected,
                model=self._ocr_model,
                extract_tables=False,
                extract_charts=ocr_extract_charts,
                extract_infographics=self._extract_infographics,
            )
        else:
            ocred = ocr_page_elements(
                detected,
                model=self._ocr_model,
                extract_tables=self._extract_tables,
                extract_charts=ocr_extract_charts,
                extract_infographics=self._extract_infographics,
            )
        if self._embed_granularity == "page":
            prepared = collapse_content_to_page_rows(
                ocred, text_column=str(self._embed_kwargs["text_column"]), modality=self._embed_modality
            )
        else:
            prepared = explode_content_to_rows(ocred, text_column=str(self._embed_kwargs["text_column"]))
        embedded = embed_text_main_text_embed(prepared, model=self._embed_model, **self._embed_kwargs)
        return embedded


class FusedIngestor(BatchIngestor):
    RUN_MODE = "fused"
    _fused_extract_flags: Dict[str, Any] = {}

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "FusedIngestor":
        """
        Configure extraction for fused execution.

        In fused mode, PDF split/extract are separate CPU stages, but all model
        stages (page-elements, OCR, embedding) run in one actor in `.embed()`.
        """
        resolved = params or ExtractParams(**kwargs)
        if params is not None and kwargs:
            resolved = params.model_copy(update=kwargs)
        kwargs = {
            **resolved.model_dump(mode="python", exclude={"remote_retry", "batch_tuning"}, exclude_none=True),
            **resolved.remote_retry.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
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
            "use_table_structure": bool(kwargs.get("use_table_structure", False)),
            "table_output_format": str(kwargs.get("table_output_format", "pseudo_markdown")),
            "extract_charts": bool(kwargs.get("extract_charts", False)),
            "extract_infographics": bool(kwargs.get("extract_infographics", False)),
            "use_graphic_elements": bool(kwargs.get("use_graphic_elements", False)),
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

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "FusedIngestor":
        """
        Run page-elements + OCR + explode + embed in one GPU actor stage.

        `fused` mode intentionally does not support remote NIM invocation.
        When _pipeline_type == "audio", uses explode + _BatchEmbedActor (no PDF stages).
        """
        resolved = params or EmbedParams(**kwargs)
        if params is not None and kwargs:
            resolved = params.model_copy(update=kwargs)
        kwargs = {
            **resolved.model_dump(
                mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True
            ),
            **resolved.runtime.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
            **resolved.fused_tuning.model_dump(mode="python", exclude_none=True),
        }
        _assert_no_remote_endpoints(dict(kwargs), context="embed")

        if getattr(self, "_pipeline_type", None) == "audio":
            embed_workers = int(kwargs.get("embed_workers", 1))
            embed_batch_size = int(kwargs.get("embed_batch_size", 256))
            embed_cpus_per_actor = float(kwargs.get("embed_cpus_per_actor", 1))
            self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=256)
            _row_fn = collapse_content_to_page_rows if resolved.embed_granularity == "page" else explode_content_to_rows
            self._rd_dataset = self._rd_dataset.map_batches(
                _row_fn,
                batch_size=embed_batch_size,
                batch_format="pandas",
                num_cpus=1,
                num_gpus=0,
            )
            self._rd_dataset = self._rd_dataset.map_batches(
                _BatchEmbedActor,
                batch_size=embed_batch_size,
                batch_format="pandas",
                num_cpus=embed_cpus_per_actor,
                num_gpus=getattr(self, "_gpu_embed", 1.0),
                compute=rd.ActorPoolStrategy(size=embed_workers),
                fn_constructor_kwargs={"params": resolved},
            )
            return self

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
