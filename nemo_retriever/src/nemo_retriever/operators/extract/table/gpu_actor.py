# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import nvtx
import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.ocr.config import resolve_ocr_v2_lang
from nemo_retriever.common.modality.table.shared import (
    table_structure_ocr_page_elements,
)


class TableStructureActor(AbstractOperator, GPUOperator):
    """
    Ray-friendly callable that initializes the table-structure model once
    per actor and runs the structure stage.
    """

    @nvtx.annotate("nrl.batch::table_structure.startup", color="blue")
    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        ocr_version: str = "v2",
        ocr_lang: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table_output_format: Optional[str] = None,
        request_timeout_s: float = 120.0,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._table_structure_invoke_url = (
            str(table_structure_invoke_url or "").strip() or str(invoke_url or "").strip()
        )
        self._ocr_invoke_url = str(ocr_invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._table_output_format = table_output_format
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )

        if self._table_structure_invoke_url:
            self._table_structure_model = None
        else:
            from nemo_retriever.models.local import NemotronTableStructureV1
            from nemo_retriever.models.warmup_registry import get_warmed_model

            warmed = get_warmed_model("table_structure")
            self._table_structure_model = warmed if warmed is not None else NemotronTableStructureV1()

        if self._ocr_invoke_url:
            self._ocr_model = None
        else:
            from nemo_retriever.models.local import NemotronOCRV2
            from nemo_retriever.models.warmup_registry import get_warmed_model

            warmed = get_warmed_model("ocr")
            if warmed is not None:
                self._ocr_model = warmed
            else:
                lang = resolve_ocr_v2_lang(ocr_version, ocr_lang)
                self._ocr_model = NemotronOCRV2(lang=lang)

        if self._table_structure_invoke_url or self._ocr_invoke_url:
            self._nim_client = NIMClient(
                max_pool_workers=int(remote_max_pool_workers),
            )
        else:
            self._nim_client = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::table_structure.batch", color="blue")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return table_structure_ocr_page_elements(
            data,
            table_structure_model=self._table_structure_model,
            table_structure_invoke_url=self._table_structure_invoke_url,
            api_key=self._api_key,
            table_output_format=self._table_output_format,
            request_timeout_s=self._request_timeout_s,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
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
                payload = {
                    "timing": None,
                    "error": {
                        "stage": "table_structure_actor_call",
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["table_structure_ocr_v1"] = [payload for _ in range(n)]
                out["table_structure_v1_num_detections"] = [0 for _ in range(n)]
                out["table_structure_v1_counts_by_label"] = [{} for _ in range(n)]
                return out
            return [
                {
                    "table_structure_ocr_v1": {
                        "timing": None,
                        "error": {
                            "stage": "table_structure_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
