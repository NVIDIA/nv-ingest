# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph operator for remote-only text embeddings."""

from __future__ import annotations

from typing import Any

import nvtx

from nemo_retriever.common.api.util.string_processing import prepend_model_provider_prefix
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.models.inference.embedding_input import ensure_embedding_input_policy_for_batch
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
from nemo_retriever.models.inference.shared import build_embed_kwargs
from nemo_retriever.models.nim.probe import probe_endpoint
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class _BatchEmbedCPUActor(AbstractOperator, CPUOperator):
    """CPU-only embedding actor that always targets a remote endpoint."""

    DEFAULT_EMBED_INVOKE_URL = "https://integrate.api.nvidia.com/v1/embeddings"

    @nvtx.annotate("nrl.batch::embedding.startup", color="blue")
    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        self._params = params
        self._kwargs = build_embed_kwargs(params)
        if "embedding_endpoint" not in self._kwargs:
            self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url") or self.DEFAULT_EMBED_INVOKE_URL

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if not endpoint:
            self._kwargs["embedding_endpoint"] = self.DEFAULT_EMBED_INVOKE_URL
            endpoint = self.DEFAULT_EMBED_INVOKE_URL
        self._model = None

        api_key = self._kwargs.get("api_key")
        if endpoint.rstrip("/") == self.DEFAULT_EMBED_INVOKE_URL and not api_key:
            raise RuntimeError(
                "CPU-only ingest uses NVIDIA's hosted embedding endpoint and requires "
                "NVIDIA_API_KEY or NGC_API_KEY. Set one of those variables, or pass "
                "--embed-invoke-url for a different embedding endpoint."
            )
        if endpoint and api_key:
            # Probe the /embeddings path with a model-name-only body — auth is
            # checked before body validation so a bad key returns 401 without
            # triggering inference. A valid key with an empty input returns 400.
            model_name = prepend_model_provider_prefix(
                self._kwargs.get("model_name"), self._kwargs.get("embed_model_provider_prefix")
            )
            probe_url = (
                endpoint if endpoint.rstrip("/").endswith("/embeddings") else endpoint.rstrip("/") + "/embeddings"
            )
            probe_endpoint(
                endpoint,
                name="embed",
                prefix="_BatchEmbedCPUActor",
                api_key=api_key,
                post_url=probe_url,
                post_body={"input": [], "model": model_name},
            )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::embedding.batch", color="blue")
    def process(self, data: Any, **kwargs: Any) -> Any:
        ensure_embedding_input_policy_for_batch(self._kwargs, data)
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
