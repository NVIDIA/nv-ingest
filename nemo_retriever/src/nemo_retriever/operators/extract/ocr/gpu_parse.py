# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.models.nim.nim import NIMClient
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.modality.ocr.shared import _error_payload, nemotron_parse_page_elements


class NemotronParseActor(AbstractOperator, GPUOperator):
    """Ray-friendly callable that initializes Nemotron Parse once per actor."""

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>",
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
    ) -> None:
        super().__init__()
        self._invoke_url = str(nemotron_parse_invoke_url or "").strip() or str(invoke_url or "").strip()
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        if self._invoke_url:
            self._model = None
            self._nim_client = NIMClient(
                max_pool_workers=int(remote_max_pool_workers),
            )
        else:
            from nemo_retriever.models.local import NemotronParseV12

            self._model = NemotronParseV12(task_prompt=self._task_prompt)
            self._nim_client = None
        self._extract_text = bool(extract_text)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return nemotron_parse_page_elements(
            data,
            model=self._model,
            invoke_url=self._invoke_url,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            task_prompt=self._task_prompt,
            extract_text=self._extract_text,
            extract_tables=self._extract_tables,
            extract_charts=self._extract_charts,
            extract_infographics=self._extract_infographics,
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
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=exc)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=exc)}]
