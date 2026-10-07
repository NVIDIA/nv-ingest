# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import nvtx
import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.models.nim.probe import probe_endpoint
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.table.shared import table_structure_ocr_page_elements

logger = logging.getLogger(__name__)


class TableStructureCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`TableStructureActor`.

    Uses NVIDIA's hosted table-structure service by default. OCR is not turned on
    unless you provide an ``ocr_invoke_url``.
    """

    DEFAULT_TABLE_STRUCTURE_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"

    @nvtx.annotate("nrl.batch::table_structure.startup", color="blue")
    def __init__(
        self,
        *,
        table_structure_invoke_url: Optional[str] = None,
        ocr_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        table_output_format: Optional[str] = None,
        request_timeout_s: float = 120.0,
        inference_batch_size: int = 8,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._table_structure_invoke_url = (
            str(table_structure_invoke_url or "").strip()
            or str(invoke_url or "").strip()
            or self.DEFAULT_TABLE_STRUCTURE_INVOKE_URL
        )
        self._ocr_invoke_url = str(ocr_invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._inference_batch_size = int(inference_batch_size)
        self._table_output_format = table_output_format
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._table_structure_model = None
        self._ocr_model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(remote_max_pool_workers),
        )

        logger.info(
            "TableStructureCPUActor initialized: table_structure_url=%s, ocr_url=%s",
            self._table_structure_invoke_url,
            self._ocr_invoke_url,
        )
        probe_endpoint(
            self._table_structure_invoke_url,
            name="table-structure",
            prefix="TableStructureCPUActor",
            api_key=self._api_key,
        )
        if self._ocr_invoke_url:
            probe_endpoint(
                self._ocr_invoke_url,
                name="ocr",
                prefix="TableStructureCPUActor",
                api_key=self._api_key,
            )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::table_structure.batch", color="blue")
    def process(self, data: Any, **kwargs: Any) -> Any:
        n_rows = len(data) if hasattr(data, "__len__") else "?"
        logger.info("TableStructureCPUActor.process: received batch of %s rows", n_rows)
        t0 = time.perf_counter()
        result = table_structure_ocr_page_elements(
            data,
            table_structure_model=self._table_structure_model,
            table_structure_invoke_url=self._table_structure_invoke_url,
            api_key=self._api_key,
            table_output_format=self._table_output_format,
            request_timeout_s=self._request_timeout_s,
            inference_batch_size=self._inference_batch_size,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        logger.info("TableStructureCPUActor.process: finished batch of %s rows in %.2fs", n_rows, elapsed)
        return result

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
                        "stage": "table_structure_cpu_actor_call",
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
                            "stage": "table_structure_cpu_actor_call",
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    }
                }
            ]
