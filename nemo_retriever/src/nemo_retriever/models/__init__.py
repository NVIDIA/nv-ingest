# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_retriever.models.embed_model_spec import (
    EmbedModelSpec,
    resolve_embed_model_spec,
    validate_embed_model_backend,
)

if TYPE_CHECKING:
    from nemo_retriever.models.model import BaseModel

NEMOTRON_3_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
NEMOTRON_3_EMBED_BF16_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
NEMOTRON_3_EMBED_NVFP4_MODEL = "nvidia/Nemotron-3-Embed-1B-NVFP4"
VL_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"

_VL_RERANK_MODEL_IDS = frozenset(
    {
        VL_RERANK_MODEL,
        "llama-nemotron-rerank-vl-1b-v2",
    }
)

# Short name to canonical model ID.
_EMBED_MODEL_ALIASES: dict[str, str] = {
    "nemotron-3-embed-1b": NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_3_EMBED_BF16_MODEL: NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_3_EMBED_NVFP4_MODEL: NEMOTRON_3_EMBED_MODEL,
    "nemo_retriever_v1": "nvidia/llama-nemotron-embed-1b-v2",
    "llama-nemotron-embed-vl-1b-v2": VL_EMBED_MODEL,
    "llama-3.2-nemoretriever-1b-vlm-embed-v1": VL_EMBED_MODEL,
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": VL_EMBED_MODEL,
}

_DEFAULT_EMBED_MODEL = NEMOTRON_3_EMBED_MODEL


def resolve_embed_model(model_name: str | None) -> str:
    """Resolve a model name or alias to its canonical ID.

    Returns ``_DEFAULT_EMBED_MODEL`` when *model_name* is ``None`` or empty.
    """
    if not model_name:
        return _DEFAULT_EMBED_MODEL
    return _EMBED_MODEL_ALIASES.get(model_name, model_name)


def is_vl_embed_model(model_name: str | None) -> bool:
    """Return True when a legacy model ID or alias names the default VL embedder."""
    return resolve_embed_model(model_name) == VL_EMBED_MODEL


def _cuda_supports_native_nvfp4() -> bool:
    """Return whether every visible CUDA device has native NVFP4 support."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return False
        return all(torch.cuda.get_device_capability(index)[0] >= 10 for index in range(torch.cuda.device_count()))
    except (ImportError, RuntimeError, AssertionError):
        return False


def resolve_local_embed_model(model_name: str | None, *, backend: str) -> str:
    """Resolve an embedding model to a backend-compatible checkpoint.

    Args:
        model_name: Logical embedding model name or alias. An omitted or empty
            value uses the default embedding model.
        backend: Normalized local inference backend, such as ``"vllm"`` or
            ``"hf"``.

    Returns:
        The resolved Hugging Face checkpoint identifier. The logical Nemotron 3
        default resolves to NVFP4 for vLLM when every visible CUDA device has
        native NVFP4 support, and to BF16 otherwise. Other model identifiers
        are returned after alias resolution.
    """
    resolved = resolve_embed_model(model_name)
    if resolved != NEMOTRON_3_EMBED_MODEL:
        return resolved
    if backend == "vllm" and _cuda_supports_native_nvfp4():
        return NEMOTRON_3_EMBED_NVFP4_MODEL
    return NEMOTRON_3_EMBED_BF16_MODEL


def is_vl_rerank_model(model_name: str | None) -> bool:
    """Return True if *model_name* refers to the VL reranker model."""
    return (model_name or "") in _VL_RERANK_MODEL_IDS


def create_local_embedder(
    model_name: str | None = None,
    *,
    backend: str = "vllm",
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
    query_max_length: int = 128,
    revision: str | None = None,
) -> Any:
    """Create the appropriate local embedding model (VL or non-VL).

    *backend* must be ``"vllm"`` or ``"hf"``.

    For non-VL models:

    - ``backend="vllm"`` (default): vLLM via ``LlamaNemotronEmbed1BV2Embedder``.
    - ``backend="hf"``: HuggingFace via ``LlamaNemotronEmbed1BV2HFEmbedder``.

    For VL models:

    - ``backend="vllm"`` (default): vLLM via ``LlamaNemotronEmbedVL1BV2VLLMEmbedder``.
    - ``backend="hf"``: HuggingFace via ``LlamaNemotronEmbedVL1BV2Embedder``.

    ``device`` applies only to HuggingFace paths. For vLLM paths, ``device`` is
    forwarded for compatibility but deprecated and ignored (vLLM placement is
    process-level); passing it emits ``DeprecationWarning``.

    The requested text limits are capped at the checkpoint-declared maximum
    before construction. Backend-specific runtime options are ignored by
    loaders that do not support them.

    Local checkpoints and compatible Hub fine-tunes are routed from their
    immutable config. Compatibility requires a supported dense Nemotron
    embedding architecture and average pooling. Output dimensions and text
    prefixes are derived from checkpoint metadata.
    """
    b = (backend or "vllm").strip().lower()
    if b not in ("vllm", "hf"):
        raise ValueError(f"backend must be 'vllm' or 'hf', got {backend!r}")

    model_id = resolve_local_embed_model(model_name, backend=b)
    spec = resolve_embed_model_spec(model_id, revision=revision, hf_cache_dir=hf_cache_dir)
    return _create_local_embedder_from_spec(
        spec,
        backend=b,
        device=device,
        hf_cache_dir=hf_cache_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=max_length,
        query_max_length=query_max_length,
    )


def _create_local_embedder_from_spec(
    spec: EmbedModelSpec,
    *,
    backend: str,
    device: str | None,
    hf_cache_dir: str | None,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    dimensions: int | None,
    normalize: bool,
    max_length: int,
    query_max_length: int,
) -> Any:
    """Construct from already-resolved metadata without another checkpoint lookup."""
    b = backend
    model_id = spec.model_id
    validate_embed_model_backend(spec, b)
    effective_max_length = min(int(max_length), spec.max_input_tokens) if spec.max_input_tokens else int(max_length)
    effective_query_max_length = (
        min(int(query_max_length), spec.max_input_tokens) if spec.max_input_tokens else int(query_max_length)
    )

    if spec.family == "vl":
        if b == "hf":
            from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            return LlamaNemotronEmbedVL1BV2Embedder(
                device=device,
                hf_cache_dir=hf_cache_dir,
                model_id=model_id,
                revision=spec.revision,
                output_dimension=spec.output_dimension,
                max_length=effective_max_length,
            )

        from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import (
            LlamaNemotronEmbedVL1BV2VLLMEmbedder,
        )

        return LlamaNemotronEmbedVL1BV2VLLMEmbedder(
            model_id=model_id,
            device=device,
            hf_cache_dir=hf_cache_dir,
            revision=spec.revision,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            normalize=normalize,
            output_dimension=spec.output_dimension,
            query_prefix=spec.query_prefix,
            document_prefix=spec.document_prefix,
        )

    if b == "hf":
        from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_hf_embedder import (
            LlamaNemotronEmbed1BV2HFEmbedder,
        )

        return LlamaNemotronEmbed1BV2HFEmbedder(
            device=device,
            hf_cache_dir=hf_cache_dir,
            normalize=normalize,
            max_length=effective_max_length,
            query_max_length=effective_query_max_length,
            model_id=model_id,
            revision=spec.revision,
            query_prefix=spec.query_prefix,
            document_prefix=spec.document_prefix,
        )

    from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder import (
        LlamaNemotronEmbed1BV2Embedder,
    )

    return LlamaNemotronEmbed1BV2Embedder(
        model_id=model_id,
        hf_cache_dir=hf_cache_dir,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=effective_max_length,
        revision=spec.revision,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
    )


_LOCAL_QUERY_BACKENDS = frozenset({"hf", "vllm"})
_LOCAL_RERANKER_BACKENDS = frozenset({"hf", "vllm"})
_LOCAL_INGEST_EMBED_BACKENDS = frozenset({"hf", "vllm"})


def normalize_backend(value: str | None, valid: frozenset[str], *, field_name: str, default: str) -> str:
    """Normalize *value* (strip + lowercase) and validate against *valid*.

    Raises ``ValueError`` referencing *field_name* on invalid input.
    Falsy *value* is replaced by *default* before validation.
    """
    v = (value or default).strip().lower()
    if v not in valid:
        raise ValueError(f"{field_name} must be one of {sorted(valid)}; got {value!r}")
    return v


def create_local_query_embedder(
    model_name: str | None = None,
    *,
    backend: str = "vllm",
    device: str | None = None,
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    dimensions: int | None = None,
    normalize: bool = True,
    max_length: int = 8192,
    query_max_length: int = 128,
    revision: str | None = None,
) -> Any:
    """Create a local embedder for *query* vectors in retrieval (Retriever / recall).

    *backend* must be ``"vllm"`` (default) or ``"hf"``.

    - ``backend="hf"``: HuggingFace for both VL and non-VL models.
    - ``backend="vllm"``: vLLM for both VL and non-VL models.

    Model architecture and quantization requirements are resolved from the
    checkpoint config; see :func:`create_local_embedder`.
    """
    b = normalize_backend(backend, _LOCAL_QUERY_BACKENDS, field_name="backend", default="vllm")

    return create_local_embedder(
        model_name,
        backend=b,
        device=device,
        hf_cache_dir=hf_cache_dir,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        dimensions=dimensions,
        normalize=normalize,
        max_length=int(max_length),
        query_max_length=int(query_max_length),
        revision=revision,
    )


def create_local_reranker(
    model_name: str | None = None,
    *,
    device: str | None = None,
    hf_cache_dir: str | None = None,
    backend: str = "vllm",
    gpu_memory_utilization: float = 0.5,
) -> "BaseModel":
    """Create the appropriate local reranker model (VL or text-only).

    Dispatches to ``NemotronRerankVLV2VLLM`` (default) or
    ``NemotronRerankVLV2`` when *model_name* matches a VL reranker ID,
    depending on *backend*.  Otherwise returns the text-only
    ``NemotronRerankV2``.

    Parameters
    ----------
    backend:
        ``"vllm"`` (default) uses vLLM's pooling runner for the VL
        reranker.  ``"hf"`` uses HuggingFace
        ``AutoModelForSequenceClassification``.  Only affects VL reranker
        dispatch; the text-only reranker always uses HuggingFace.
    gpu_memory_utilization:
        Fraction of GPU memory for the vLLM engine (only used when
        *backend* is ``"vllm"``).
    """
    b = normalize_backend(backend, _LOCAL_RERANKER_BACKENDS, field_name="backend", default="vllm")
    if is_vl_rerank_model(model_name):
        if b == "vllm":
            from nemo_retriever.models.local.nemotron_rerank_vl_v2 import NemotronRerankVLV2VLLM

            return NemotronRerankVLV2VLLM(
                model_name=model_name,
                device=device,
                hf_cache_dir=hf_cache_dir,
                gpu_memory_utilization=gpu_memory_utilization,
            )

        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        return NemotronRerankVLV2(
            model_name=model_name,
            device=device,
            hf_cache_dir=hf_cache_dir,
        )

    from nemo_retriever.models.local.nemotron_rerank_v2 import NemotronRerankV2

    return NemotronRerankV2(
        model_name=model_name or "nvidia/llama-nemotron-rerank-1b-v2",
        device=device,
        hf_cache_dir=hf_cache_dir,
    )


_LOCAL_AGENT_LLM_BACKENDS = frozenset({"vllm"})


def create_local_agent_llm(
    model_name: str,
    *,
    backend: str = "vllm",
    hf_cache_dir: str | None = None,
    gpu_memory_utilization: float = 0.8,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
) -> Any:
    """Create a local agent LLM chat-completion callable owned by the caller.

    The callable mirrors ``invoke_chat_completion_step`` and returns an
    OpenAI-compatible chat-completions response dict. V1 supports the in-process
    vLLM backend and uses process-level vLLM placement (for example,
    ``CUDA_VISIBLE_DEVICES`` plus ``tensor_parallel_size``). Callers (typically
    ``AgenticRetriever``) should reuse one instance for a CLI job and
    call ``unload()`` when the job finishes.
    """

    b = normalize_backend(backend, _LOCAL_AGENT_LLM_BACKENDS, field_name="backend", default="vllm")
    if b == "vllm":
        from nemo_retriever.models.local.agent_llm import LocalAgentLLMConfig, create_vllm_agent_chat_llm

        return create_vllm_agent_chat_llm(
            LocalAgentLLMConfig(
                model_path=model_name,
                hf_cache_dir=hf_cache_dir,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        )

    raise ValueError(f"Unsupported local agent LLM backend {backend!r}")
