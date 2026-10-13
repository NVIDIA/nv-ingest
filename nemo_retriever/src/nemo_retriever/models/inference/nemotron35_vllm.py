# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-safe registration and attention construction for N3.5 retrieval."""

NEMOTRON35_VLLM_ARCHITECTURE = "NRLNemotron35EmbeddingModel"


def register_nemotron35_vllm() -> None:
    """Register only the NRL architecture, without importing CUDA model code."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        NEMOTRON35_VLLM_ARCHITECTURE,
        "nemo_retriever.models.local.nemotron35_vllm:NRLNemotron35EmbeddingModel",
    )


def create_bidirectional_attention(model, *, attention_cls, attention_type, pp_indices):
    """Construct noncausal text attention without changing vendor model classes."""
    text = model.text_config
    if (
        model.config.model_type != "mistral3"
        or text.model_type != "ministral3"
        or getattr(text, "is_causal", None) is not False
    ):
        raise ValueError("The N3.5 vLLM adapter requires a bidirectional Mistral3 retrieval checkpoint.")
    heads = model.model_config.get_num_attention_heads(model.parallel_config)
    size = model.model_config.get_head_size()
    kv_heads = model.model_config.get_num_kv_heads(model.parallel_config)
    start, end = pp_indices(text.num_hidden_layers, model.pp_group.rank_in_group, model.pp_group.world_size)
    layers = getattr(text, "layer_types", None)
    return {
        index: attention_cls(
            num_heads=heads,
            head_size=size,
            scale=size**-0.5,
            num_kv_heads=kv_heads,
            cache_config=model.cache_config,
            quant_config=model.quant_config,
            logits_soft_cap=getattr(text, "attn_logit_softcapping", None),
            per_layer_sliding_window=(
                getattr(text, "sliding_window", None) if layers and layers[index] == "sliding_attention" else None
            ),
            prefix=f"{index}.attn",
            attn_type=attention_type,
        )
        for index in range(start, end)
    }
