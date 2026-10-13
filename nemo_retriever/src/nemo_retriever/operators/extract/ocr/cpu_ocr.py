# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import nvtx
import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.ocr.shared import _error_payload
from nemo_retriever.common.modality.ocr.shared import ocr_page_elements


class OCRCPUActor(AbstractOperator, CPUOperator):
    """Remote OCR variant of :class:`OCRActor`.

    Defaults to the hosted Nemotron OCR v2 NIM endpoint.
    """

    DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"

    @nvtx.annotate("nrl.batch::ocr.startup", color="blue")
    def __init__(self, **ocr_kwargs: Any) -> None:
        super().__init__(**ocr_kwargs)
        self.ocr_kwargs = dict(ocr_kwargs)
        invoke_url = (
            str(self.ocr_kwargs.get("ocr_invoke_url") or "").strip()
            or str(self.ocr_kwargs.get("invoke_url") or "").strip()
            or self.DEFAULT_INVOKE_URL
        )
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
        self._model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(self._remote_retry.remote_max_pool_workers),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::ocr.batch", color="blue")
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
                payload = _error_payload(stage="cpu_actor_call", exc=exc)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["ocr"] = [payload for _ in range(n)]
                out["ocr_v1_num_detections"] = [0 for _ in range(n)]
                out["ocr_v1_counts_by_label"] = [{} for _ in range(n)]
                return out
            return [{"ocr": _error_payload(stage="cpu_actor_call", exc=exc)}]


OCRV2CPUActor = OCRCPUActor
