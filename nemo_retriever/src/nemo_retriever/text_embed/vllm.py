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

Troubleshooting enforce_eager=False (torch.compile):
  If you see "failed to map segment from shared object", the cache dir is on a
  filesystem that does not allow executing shared libs (e.g. noexec mount). /tmp
  and /dev/shm are often noexec on Linux. We auto-pick a dir under /run/user/<uid>
  (if present), then ~/.cache/vllm/torch_compile_inductor, then /dev/shm. Set
  compile_cache_dir to a path on a non-noexec filesystem if needed, or use
  enforce_eager=True.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

VLLM_DTYPE = "bfloat16"
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"


def _default_compile_cache_dir() -> Optional[str]:
    """Return a cache dir for torch inductor/triton that supports loading shared libs (no noexec).
    Prefer /run/user/<uid> (often exec-enabled tmpfs), then ~/.cache, then /dev/shm.
    Avoid /tmp and /dev/shm when they are mounted noexec (common on Linux)."""
    candidates: List[tuple[str, str]] = []
    uid = os.getuid() if hasattr(os, "getuid") else 0
    run_user = f"/run/user/{uid}"
    if os.path.isdir(run_user) and os.access(run_user, os.W_OK):
        candidates.append((run_user, os.path.join(run_user, "vllm_torch_compile")))
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    vllm_cache = os.path.join(cache_home, "vllm", "torch_compile_inductor")
    if cache_home:
        candidates.append((cache_home, vllm_cache))
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        candidates.append((shm, os.path.join(shm, f"vllm_torch_compile_{uid}")))
    for _parent, path in candidates:
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
            return path
        except OSError:
            continue
    return None


def create_vllm_llm(
    model: str,
    *,
    dimensions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    compile_cache_dir: Optional[str] = None,
) -> Any:
    """
    Create and return a vLLM LLM instance for embedding (pooling runner).
    Caller can reuse it across many embed batches to avoid repeated model load and CUDA graph capture.

    Uses bfloat16 and FLASH_ATTN backend (fixed for this module).
    """
    try:
        from vllm import LLM
    except ImportError as e:
        raise RuntimeError(
            "vLLM embedding requires the embed-vllm extra. "
            "Install with: uv pip install -e '.[embed-vllm]' or pip install -e '.[embed-vllm]'"
        ) from e

    if not enforce_eager:
        cache_dir = compile_cache_dir if compile_cache_dir is not None else _default_compile_cache_dir()
        if cache_dir:
            os.makedirs(cache_dir, mode=0o700, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            os.environ["TRITON_CACHE_DIR"] = cache_dir
            logger.debug("vLLM: using compile cache dir %s", cache_dir)

    pooler_config = None
    try:
        from vllm.config.pooler import PoolerConfig

        try:
            pooler_config = PoolerConfig(seq_pooling_type="MEAN", dimensions=dimensions)
        except TypeError:
            pooler_config = PoolerConfig(pooling_type="MEAN", dimensions=dimensions)
    except Exception:
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
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if pooler_config is not None:
        kwargs["pooler_config"] = pooler_config

    return LLM(**kwargs)


def embed_with_vllm_llm(
    prompts: List[str],
    llm: Any,
    *,
    batch_size: int = 256,
    prefix: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute embeddings using an existing vLLM LLM instance (no new model load).
    Use this when the caller holds a shared LLM (e.g. one per Ray actor).
    """
    if prefix:
        prompts = [str(prefix) + p for p in prompts]
    if not prompts:
        return []

    all_embeddings: List[List[float]] = []
    for i in range(0, len(prompts), max(1, batch_size)):
        batch = prompts[i : i + max(1, batch_size)]
        outputs = llm.embed(batch)
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


def embed_via_vllm(
    prompts: List[str],
    *,
    model: str,
    batch_size: int = 256,
    prefix: Optional[str] = None,
    dimensions: Optional[int] = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.45,
    enforce_eager: bool = False,
    compile_cache_dir: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute embeddings via vLLM's Python API (no server).
    Uses bfloat16 and FLASH_ATTN backend (fixed for this module).

    Parameters
    ----------
    prompts : list of str
        Texts to embed. If prefix is set, each prompt is prefixed (e.g. "query: " or "passage: ").
    model : str
        HuggingFace model id (e.g. nvidia/llama-3.2-nv-embedqa-1b-v2) or local path.
    batch_size : int
        Max prompts per llm.embed() call (vLLM may batch internally as well).
    prefix : str, optional
        Prefix to prepend to each prompt (e.g. "passage: ", "query: ").
    dimensions : int, optional
        Optional embedding dimension (Matryoshka); if supported by model and vLLM.
    tensor_parallel_size : int
        Number of GPUs for tensor parallelism.
    trust_remote_code : bool
        Passed to LLM when loading the model.
    max_model_len : int, optional
        Max sequence length; default uses model config.
    gpu_memory_utilization : float
        Fraction of GPU memory for vLLM (default 0.45).
    enforce_eager : bool
        If False (default), use torch.compile/CUDAGraphs for speed. If True, disable for stability.
    compile_cache_dir : str, optional
        Directory for PyTorch inductor/Triton JIT cache.

    Returns
    -------
    list of list of float
        One embedding vector per prompt, in order.
    """
    llm = create_vllm_llm(
        model,
        dimensions=dimensions,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        compile_cache_dir=compile_cache_dir,
    )
    return embed_with_vllm_llm(prompts, llm, batch_size=batch_size, prefix=prefix)


__all__ = ["create_vllm_llm", "embed_via_vllm", "embed_with_vllm_llm"]
