# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped bidirectional multimodal pooling model for the pinned vLLM runtime."""

from vllm.model_executor.models.transformers import TransformersMultiModalEmbeddingModel
from vllm.model_executor.models.transformers.base import (
    AttentionType,
    EncoderOnlyAttention,
    get_pp_indices,
)

from nemo_retriever.models.inference.nemotron35_vllm import (
    create_bidirectional_attention,
)


class NRLNemotron35EmbeddingModel(TransformersMultiModalEmbeddingModel):
    default_seq_pooling_type = "MEAN"

    def extra_repr(self):
        attention_classes = sorted({type(layer).__name__ for layer in self.attention_instances.values()})
        return f"text_is_causal={self.text_config.is_causal}, attention={','.join(attention_classes)}"

    def create_attention_instances(self):
        return create_bidirectional_attention(
            self,
            attention_cls=EncoderOnlyAttention,
            attention_type=AttentionType.ENCODER_ONLY,
            pp_indices=get_pp_indices,
        )
