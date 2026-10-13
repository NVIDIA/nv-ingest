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
    """Mean-pool a bidirectional Mistral3 text backbone with native image processing."""

    default_seq_pooling_type = "MEAN"

    def extra_repr(self) -> str:
        """Return checkpoint causality and constructed attention classes for inspection.

        Parameters
        ----------
        None
            Uses the initialized model's text configuration and attention instances.

        Returns
        -------
        str
            Compact text-attention description for the PyTorch model representation.

        Raises
        ------
        AttributeError
            If called before text configuration or attention instances are initialized.
        """
        attention_classes = sorted({type(layer).__name__ for layer in self.attention_instances.values()})
        return f"text_is_causal={self.text_config.is_causal}, attention={','.join(attention_classes)}"

    def create_attention_instances(self) -> dict[int, EncoderOnlyAttention]:
        """Create shard-local encoder-only attention from checkpoint configuration.

        Parameters
        ----------
        None
            Uses this model's initialized backend, cache, and parallel settings.

        Returns
        -------
        dict[int, Any]
            Global text-layer indices mapped to EncoderOnlyAttention instances.

        Raises
        ------
        ValueError
            If the checkpoint does not request bidirectional Mistral3/Ministral3 attention.
        RuntimeError
            If backend attention construction fails.
        """
        return create_bidirectional_attention(
            self,
            attention_cls=EncoderOnlyAttention,
            attention_type=AttentionType.ENCODER_ONLY,
            pp_indices=get_pp_indices,
        )
