# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM batched embedding inference.

Uses vLLM's Python API (LLM with runner="pooling" and llm.embed())
to compute embeddings without running a vLLM server. Use this when you want
the same embedding model (e.g. nvidia/llama-3.2-nv-embedqa-1b-v2) with vLLM's batched
inference and no HTTP server.

Uses bfloat16 and FLASH_ATTN backend by default for best throughput.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from nemo_retriever.common.schemas.embedding import format_embedding_input

logger = logging.getLogger(__name__)

VLLM_DTYPE = "bfloat16"
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"
VLLM_DEEP_GEMM_WARMUP_DEFAULT = "skip"

# Multi-GPU collectives that rely on NVLink multicast (NCCL NVLS and torch
# symmetric-memory all-reduce) abort with "NCCL error: unhandled cuda error"
# during engine startup when the visible GPUs are only PCIe-connected, which is
# the common two-workstation-card layout. Falling back to ring collectives costs
# throughput that these hosts cannot use anyway.
VLLM_NO_NVLINK_ENV_DEFAULTS = {
    "NCCL_NVLS_ENABLE": "0",
    "TORCH_SYMM_MEM_DISABLE_MULTICAST": "1",
}


def apply_vllm_startup_defaults(*, tensor_parallel_size: int = 1) -> None:
    """Apply conservative vLLM startup defaults without overriding users."""

    # DeepGEMM can still be used by vLLM at runtime. This only skips the
    # ahead-of-time warmup path, which may fail before local inference starts
    # when the optional DeepGEMM/CUDA-toolkit stack is not discoverable.
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", VLLM_DEEP_GEMM_WARMUP_DEFAULT)

    tp = int(tensor_parallel_size)
    if tp > 1 and not nvlink_is_available(tensor_parallel_size=tp):
        for name, value in VLLM_NO_NVLINK_ENV_DEFAULTS.items():
            os.environ.setdefault(name, value)
        logger.info(
            "No NVLink detected in the tensor-parallel GPU group; running tensor_parallel_size=%d with %s "
            "so vLLM does not start NVLink multicast collectives.",
            tp,
            ", ".join(f"{name}={value}" for name, value in VLLM_NO_NVLINK_ENV_DEFAULTS.items()),
        )


def _visible_nvml_device_indices(device_count: int) -> list[int] | None:
    """Map ``CUDA_VISIBLE_DEVICES`` to physical NVML indices.

    NVML always enumerates every physical GPU, so scanning the whole host would
    misclassify a PCIe-only visible pair on a machine that also has an
    NVLink-connected pair. ``None`` means the selection is not resolvable here
    (UUID or MIG tokens) and callers treat it as unknown connectivity.
    """

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return list(range(device_count))

    indices: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if not token.isdigit() or int(token) >= device_count:
            return None
        indices.append(int(token))
    return indices


def nvlink_is_available(*, tensor_parallel_size: int | None = None) -> bool:
    """Return whether NVML reports an active NVLink in the vLLM TP GPU group.

    When ``tensor_parallel_size`` is set, only the first N CUDA-visible devices
    are checked — matching vLLM's default rank→device mapping — so extra visible
    GPUs outside the TP group cannot mask a PCIe-only shard. An active link only
    counts when it reaches another GPU in that group (or an NVSwitch): hosts with
    separately bridged pairs give every device a link that leads elsewhere.

    Unknown counts as available so an NVML gap never silently downgrades
    collectives on a host that does have NVLink among the selected devices.
    """

    try:
        import pynvml
    except ImportError:
        return True

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("NVML unavailable; assuming NVLink is present", exc_info=True)
        return True

    try:
        device_count = int(pynvml.nvmlDeviceGetCount())
        device_indices = _visible_nvml_device_indices(device_count)
        if device_indices is None:
            logger.debug(
                "Could not resolve CUDA_VISIBLE_DEVICES=%r to NVML indices; assuming NVLink is present",
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
            return True

        if tensor_parallel_size is not None:
            tp = int(tensor_parallel_size)
            if tp < 1:
                return True
            device_indices = device_indices[:tp]

        handles = {index: pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(device_count)}
        bus_ids = {
            index: str(pynvml.nvmlDeviceGetPciInfo(handle).busId).strip().lower() for index, handle in handles.items()
        }
        tp_bus_ids = {bus_ids[index] for index in device_indices}
        host_bus_ids = set(bus_ids.values())
        max_links = int(getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 18))
        for device_index in device_indices:
            handle = handles[device_index]
            for link in range(max_links):
                try:
                    active = pynvml.nvmlDeviceGetNvLinkState(handle, link) == 1
                except pynvml.NVMLError:
                    # Not supported on this device, or the link index is beyond
                    # what it exposes; keep probing the remaining TP devices.
                    break
                if not active:
                    continue
                try:
                    remote = str(pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link).busId).strip().lower()
                except (pynvml.NVMLError, AttributeError):
                    return True  # Cannot see the peer, so trust the active link.
                # A peer that is not one of this host's GPUs is an NVSwitch, which
                # still connects the whole tensor-parallel group.
                if remote in tp_bus_ids or remote not in host_bus_ids:
                    return True
        return False
    except pynvml.NVMLError:
        logger.debug("NVML NVLink query failed; assuming NVLink is present", exc_info=True)
        return True
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            logger.debug("Ignoring NVML shutdown error", exc_info=True)


def create_vllm_llm(
    model: str,
    *,
    revision: Optional[str] = None,
    dimensions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    hf_overrides: Optional[dict] = None,
    limit_mm_per_prompt: Optional[dict] = None,
    download_dir: Optional[str] = None,
    model_impl: Optional[str] = None,
) -> Any:
    """
    Create and return a vLLM LLM instance for embedding (pooling runner).
    Caller can reuse it across many embed batches to avoid repeated model load and CUDA graph capture.

    Uses bfloat16 and FLASH_ATTN backend (fixed for this module).

    """
    apply_vllm_startup_defaults(tensor_parallel_size=tensor_parallel_size)
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError("vLLM is not installed. Install with: uv pip install -e '.[local]'") from e

    pooler_config = None
    try:
        from vllm.config.pooler import PoolerConfig

        try:
            pooler_config = PoolerConfig(seq_pooling_type="MEAN", dimensions=dimensions)
        except TypeError:
            try:
                pooler_config = PoolerConfig(pooling_type="MEAN", dimensions=dimensions)
            except (TypeError, ValueError):
                pooler_config = PoolerConfig()
    except ImportError:
        pooler_config = None

    kwargs: dict = {
        "model": model,
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": VLLM_DTYPE,
        "runner": "pooling",
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "attention_backend": VLLM_ATTENTION_BACKEND,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if download_dir is not None:
        kwargs["download_dir"] = download_dir
    if model_impl is not None:
        kwargs["model_impl"] = model_impl
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if hf_overrides is not None:
        kwargs["hf_overrides"] = hf_overrides
    if pooler_config is not None:
        kwargs["pooler_config"] = pooler_config
    if limit_mm_per_prompt is not None:
        kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

    return LLM(**kwargs)


def _pooling_params_for_normalize(normalize: Optional[bool]) -> Any:
    """Create per-request vLLM pooling parameters when normalization is disabled."""
    if normalize is not False:
        return None
    try:
        from vllm.pooling_params import PoolingParams

        return PoolingParams(use_activation=False)
    except (ImportError, TypeError) as e:
        raise RuntimeError(
            f"Failed to create PoolingParams for normalize=False: {e}. "
            "Ensure your vLLM installation supports PoolingParams "
            "(install with: uv pip install -e '.[local]')."
        ) from e


def embed_with_vllm_llm(
    prompts: List[str],
    llm: Any,
    *,
    batch_size: int = 256,
    prefix: Optional[str] = None,
    normalize: Optional[bool] = None,
) -> List[List[float]]:
    """
    Compute embeddings using an existing vLLM LLM instance (no new model load).
    Use this when the caller holds a shared LLM (e.g. one per Ray actor).

    Args:
        prompts: Input strings to embed.
        llm: A vLLM LLM instance created with ``runner="pooling"``.
        batch_size: Number of prompts per vLLM call.
        prefix: Optional string prepended to every prompt before encoding.
        normalize: Whether to request normalized embeddings. ``False`` passes
            ``PoolingParams(use_activation=False)`` to ``llm.embed()`` to skip
            pooler activation, such as L2 normalization. ``True`` and ``None``
            omit ``PoolingParams`` and preserve vLLM's compiled defaults.
    """
    pooling_params = _pooling_params_for_normalize(normalize)

    if prefix:
        prompts = [format_embedding_input(prompt, prefix) for prompt in prompts]
    if not prompts:
        return []

    all_embeddings: List[List[float]] = []
    for i in range(0, len(prompts), max(1, batch_size)):
        batch = prompts[i : i + max(1, batch_size)]
        if pooling_params is None:
            outputs = llm.embed(batch)
        else:
            outputs = llm.embed(batch, pooling_params=pooling_params)
        for out in outputs:
            emb = getattr(getattr(out, "outputs", None), "embedding", None)
            if emb is not None:
                if hasattr(emb, "tolist"):
                    all_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_embeddings.append([float(x) for x in emb])
                else:
                    all_embeddings.append(list(emb))
            else:
                all_embeddings.append([])
    return all_embeddings


def embed_multimodal_with_vllm_llm(
    prompt_dicts: List[dict],
    llm: Any,
    *,
    batch_size: int = 64,
    normalize: Optional[bool] = None,
) -> List[List[float]]:
    """
    Compute embeddings for multimodal prompts using an existing vLLM LLM instance.

    Each element of ``prompt_dicts`` must be a dict with:
      - ``"prompt"``: text string including the ``<image>`` placeholder
      - ``"multi_modal_data"``: ``{"image": PIL.Image.Image}``

    The LLM must have been created with ``limit_mm_per_prompt={"image": 1}``.
    ``normalize=False`` disables the model pooler's normalization activation.
    Returns one embedding vector (list of floats) per input; ``[]`` for failures.
    """
    if not prompt_dicts:
        return []

    pooling_params = _pooling_params_for_normalize(normalize)
    all_embeddings: List[List[float]] = []
    for i in range(0, len(prompt_dicts), max(1, batch_size)):
        batch = prompt_dicts[i : i + max(1, batch_size)]
        if pooling_params is None:
            outputs = llm.embed(batch)
        else:
            outputs = llm.embed(batch, pooling_params=pooling_params)
        for out in outputs:
            emb = getattr(getattr(out, "outputs", None), "embedding", None)
            if emb is not None:
                if hasattr(emb, "tolist"):
                    all_embeddings.append(emb.tolist())
                elif isinstance(emb, list):
                    all_embeddings.append([float(x) for x in emb])
                else:
                    all_embeddings.append(list(emb))
            else:
                all_embeddings.append([])
    return all_embeddings


__all__ = [
    "apply_vllm_startup_defaults",
    "nvlink_is_available",
    "create_vllm_llm",
    "embed_with_vllm_llm",
    "embed_multimodal_with_vllm_llm",
]
