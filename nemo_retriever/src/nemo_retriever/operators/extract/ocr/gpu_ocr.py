# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.common.nvtx import batch_phase
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.ocr.config import resolve_ocr_v2_lang
from nemo_retriever.common.modality.ocr.shared import Image, _error_payload, ocr_page_elements


class OCRActor(AbstractOperator, GPUOperator):
    """Ray-friendly callable that initializes Nemotron OCR v2 once per actor."""

    @batch_phase("ocr.startup")
    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        import warnings

        if Image is not None:
            warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

        self.ocr_kwargs = dict(ocr_kwargs)
        invoke_url = (
            str(self.ocr_kwargs.get("ocr_invoke_url") or "").strip()
            or str(self.ocr_kwargs.get("invoke_url") or "").strip()
        )
        if invoke_url:
            self.ocr_kwargs["invoke_url"] = invoke_url

        self.ocr_kwargs["extract_text"] = bool(self.ocr_kwargs.get("extract_text", False))
        self.ocr_kwargs["extract_tables"] = bool(self.ocr_kwargs.get("extract_tables", False))
        self.ocr_kwargs["extract_charts"] = bool(self.ocr_kwargs.get("extract_charts", False))
        self.ocr_kwargs["extract_infographics"] = bool(self.ocr_kwargs.get("extract_infographics", False))
        self.ocr_kwargs["use_table_structure"] = bool(self.ocr_kwargs.get("use_table_structure", False))
        self.ocr_kwargs["request_timeout_s"] = float(self.ocr_kwargs.get("request_timeout_s", 120.0))
        self.ocr_kwargs["inference_batch_size"] = int(self.ocr_kwargs.get("inference_batch_size", 8))

        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(self.ocr_kwargs.get("remote_max_pool_workers", 16)),
            remote_max_retries=int(self.ocr_kwargs.get("remote_max_retries", 10)),
            remote_max_429_retries=int(self.ocr_kwargs.get("remote_max_429_retries", 5)),
        )
        if invoke_url:
            self._model = None
            self._nim_client = NIMClient(
                max_pool_workers=int(self._remote_retry.remote_max_pool_workers),
            )
        else:
            from nemo_retriever.models.local import NemotronOCRV2
            from nemo_retriever.models.warmup_registry import get_warmed_model

            lang = resolve_ocr_v2_lang(
                str(self.ocr_kwargs.get("ocr_version", "v2")),
                self.ocr_kwargs.get("ocr_lang"),
            )
            warmed = get_warmed_model("ocr")
            self._model = warmed if warmed is not None else NemotronOCRV2(lang=lang)
            self._nim_client = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @batch_phase("ocr.batch")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return ocr_page_elements(
            data,
            model=self._model,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **self.ocr_kwargs,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as exc:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=exc)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["ocr"] = [payload for _ in range(n)]
                out["ocr_v1_num_detections"] = [0 for _ in range(n)]
                out["ocr_v1_counts_by_label"] = [{} for _ in range(n)]
                return out
            return [{"ocr": _error_payload(stage="actor_call", exc=exc)}]


OCRV2Actor = OCRActor
