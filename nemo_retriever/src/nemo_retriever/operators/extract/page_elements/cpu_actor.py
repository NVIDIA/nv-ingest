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
from nemo_retriever.common.modality.page_elements.shared import _error_payload, detect_page_elements_v3


class PageElementDetectionCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant of :class:`PageElementDetectionActor`.

    Defaults to the build.nvidia.com endpoint for ``nemotron-page-elements-v3``.
    No local GPU model is loaded.
    """

    DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"

    @nvtx.annotate("nrl.batch::page_elements.startup", color="blue")
    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)
        self.detect_kwargs = dict(detect_kwargs)
        invoke_url = (
            str(self.detect_kwargs.get("page_elements_invoke_url") or "").strip()
            or str(self.detect_kwargs.get("invoke_url") or "").strip()
            or self.DEFAULT_INVOKE_URL
        )
        self.detect_kwargs["invoke_url"] = invoke_url
        self._model = None
        self._nim_client = NIMClient(
            max_pool_workers=int(self.detect_kwargs.get("remote_max_pool_workers", 24)),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::page_elements.batch", color="blue")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return detect_page_elements_v3(
            data,
            model=self._model,
            nim_client=self._nim_client,
            **self.detect_kwargs,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(pages_df, **override_kwargs)
        except Exception as exc:
            if isinstance(pages_df, pd.DataFrame):
                out = pages_df.copy()
                payload = _error_payload(stage="cpu_actor_call", exc=exc)
                out["page_elements_v3"] = [payload for _ in range(len(out.index))]
                out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
                out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"page_elements_v3": _error_payload(stage="cpu_actor_call", exc=exc)}]
