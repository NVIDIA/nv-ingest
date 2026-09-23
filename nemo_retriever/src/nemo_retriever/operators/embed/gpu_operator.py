# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU graph operator for embedding text and multimodal content."""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.nvtx import batch_phase
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.models.inference.embedding_input import ensure_embedding_input_policy_for_batch
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
from nemo_retriever.models.inference.shared import build_embed_kwargs
from nemo_retriever.models.local_embedder_spec import LocalEmbedderSpec
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.gpu_operator import GPUOperator


class _BatchEmbedActor(AbstractOperator, GPUOperator):
    """Graph embedding actor that loads a local embedder or calls a remote endpoint."""

    @batch_phase("embedding.startup")
    def __init__(self, params: EmbedParams) -> None:
        super().__init__()
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r".*`input_embeds` is deprecated.*create_bidirectional_mask.*",
            category=FutureWarning,
        )

        self._params = params
        self._kwargs = build_embed_kwargs(params)

        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            self._model = None
            return

        local_spec = LocalEmbedderSpec.from_config(self._kwargs)
        self._kwargs["_embedding_checkpoint"] = local_spec.checkpoint
        self._kwargs["model_name"] = local_spec.model_name
        self._kwargs["embed_model_name"] = local_spec.model_name
        self._kwargs["embed_model_revision"] = local_spec.revision
        self._kwargs["max_length"] = local_spec.max_length
        self._kwargs["query_max_length"] = local_spec.query_max_length
        self._kwargs["_embedding_prefix_if_missing"] = local_spec.prefix_if_missing

        from nemo_retriever.models.warmup_registry import get_warmed_model

        warmed = get_warmed_model("embed", expected_identity=local_spec)
        if warmed is not None:
            self._model = warmed
            return

        self._model = local_spec.create()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @batch_phase("embedding.batch")
    def process(self, data: Any, **kwargs: Any) -> Any:
        ensure_embedding_input_policy_for_batch(self._kwargs, data)
        return embed_text_main_text_embed(data, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any) -> Any:
        return self.run(batch_df)
