# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ColEmbed singleton retriever (deep module).

This module encapsulates all heavy retrieval operations for the NeMo Retriever
ColEmbed model behind a small interface:

- init(...): load model + corpus embeddings (cached) once
- retrieve(query): run retrieval for a single query string
- unload(): free GPU/CPU memory

The exported `retriever` object is a module-level singleton.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Required dependencies not installed for ColEmbed retriever. "
        "Please install at least: torch (and for actual retrieval: transformers, optionally flash-attn)."
    ) from e


class _ColEmbedState:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_scoring_batch_size: int,
        batch_size: int,
        corpus_batch_size: int,
        top_k: int,
        cache_dir: Path,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_scoring_batch_size = max_scoring_batch_size
        self.batch_size = batch_size
        self.corpus_batch_size = corpus_batch_size
        self.top_k = top_k
        self.cache_dir = cache_dir

        self.dataset_name: Optional[str] = None
        self.corpus_ids: Optional[List[str]] = None
        self.corpus_markdown: Optional[List[str]] = None
        self.corpus_embeddings_cpu: Optional[torch.Tensor] = None
        self.corpus_embeddings_gpu: Optional[torch.Tensor] = None

        self.model = self._load_model()

    def _load_model(self):
        # CUDA is required (matching the original pipeline behavior).
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. ColEmbed retriever requires an NVIDIA GPU.")

        # Compatibility shim for torch/transformers version skew.
        from retrieval_bench.utils.torch_compat import patch_torch_is_autocast_enabled

        patch_torch_is_autocast_enabled()

        # Lazy import so importing this module doesn't require transformers.
        from transformers import AutoModel  # type: ignore

        try:
            model = AutoModel.from_pretrained(
                self.model_id,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model = AutoModel.from_pretrained(
                self.model_id,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
            )

        model.eval()
        return model

    def _corpus_cache_path(self, dataset_name: str) -> Path:
        dataset_slug = dataset_name.replace("/", "__")
        model_slug = self.model_id.split("/")[-1].replace("/", "__")
        key = f"{dataset_name}::{self.model_id}"
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        filename = f"corpus_embeddings__{dataset_slug}__{model_slug}__{key_hash}.pt"
        return self.cache_dir / filename

    def _embed_corpus_batched(self, corpus: Sequence[Any]) -> torch.Tensor:
        corpus_embeddings: List[torch.Tensor] = []
        num_batches = (len(corpus) + self.corpus_batch_size - 1) // self.corpus_batch_size

        for i in range(0, len(corpus), self.corpus_batch_size):
            batch_idx = i // self.corpus_batch_size + 1
            batch = corpus[i : i + self.corpus_batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.forward_passages(batch, batch_size=len(batch))
                corpus_embeddings.append(batch_embeddings.cpu())

                del batch_embeddings
                torch.cuda.empty_cache()

            # lightweight progress marker (avoid spamming logs)
            if batch_idx % 25 == 0 or batch_idx == num_batches:
                _ = torch.cuda.memory_allocated()

        return torch.cat(corpus_embeddings, dim=0)

    def _load_or_build_corpus_embeddings(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Any],
    ) -> torch.Tensor:
        cache_path = self._corpus_cache_path(dataset_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            try:
                emb = torch.load(cache_path, map_location="cpu")
                if not isinstance(emb, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor in cache, got {type(emb)}")
                if emb.shape[0] != len(corpus_ids):
                    raise ValueError(
                        f"Cached embeddings mismatch: cached={emb.shape[0]} vs corpus_ids={len(corpus_ids)}"
                    )
                return emb
            except Exception:
                logger.debug("Cache load failed for %s, recomputing", cache_path, exc_info=True)
                # fall through to recompute
                pass

        t0 = time.time()
        emb = self._embed_corpus_batched(corpus)
        elapsed = time.time() - t0
        print(f"[cache] corpus embedding took {elapsed:.1f}s ({len(corpus)} docs)")
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(emb, tmp_path)
        os.replace(tmp_path, cache_path)
        return emb

    def _embed_query(self, query: str) -> torch.Tensor:
        with torch.no_grad():
            q_emb = self.model.forward_queries([query], batch_size=1).detach()
        return q_emb[0].to(self.device)  # [seq_len, dim] on GPU

    def _score_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        emb_gpu = self.corpus_embeddings_gpu
        emb_cpu = self.corpus_embeddings_cpu
        if emb_gpu is None and emb_cpu is None:
            raise RuntimeError("No corpus embeddings available.")

        num_corpus = (emb_gpu if emb_gpu is not None else emb_cpu).shape[0]
        device = self.device
        scores = torch.empty(num_corpus, dtype=torch.float32, device=device)

        chunk = max(1, int(self.max_scoring_batch_size))

        with torch.no_grad():
            q_t = query_embedding.transpose(0, 1)  # [dim, q_seq]

            for c_start in range(0, num_corpus, chunk):
                c_end = min(c_start + chunk, num_corpus)
                c_chunk = emb_gpu[c_start:c_end] if emb_gpu is not None else emb_cpu[c_start:c_end].to(device)
                token_sims = torch.matmul(c_chunk, q_t)  # [chunk, c_seq, q_seq]
                chunk_scores = token_sims.max(dim=1).values.float().sum(dim=1)  # [chunk]
                scores[c_start:c_end] = chunk_scores

        return scores

    def retrieve_one(
        self, query: str, *, return_markdown: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        if (
            self.corpus_ids is None
            or (self.corpus_embeddings_gpu is None and self.corpus_embeddings_cpu is None)
            or self.corpus_markdown is None
        ):
            raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")

        q_emb = self._embed_query(query)
        scores = self._score_query(q_emb)

        k = min(self.top_k, len(self.corpus_ids))
        topk_scores, topk_indices = torch.topk(scores, k)

        corpus_ids = self.corpus_ids
        topk_indices_cpu = topk_indices.cpu().tolist()
        topk_scores_cpu = topk_scores.cpu().tolist()
        run = {corpus_ids[int(idx)]: float(score) for idx, score in zip(topk_indices_cpu, topk_scores_cpu)}

        if not return_markdown:
            return run

        corpus_markdown = self.corpus_markdown
        markdown_by_id = {corpus_ids[int(idx)]: corpus_markdown[int(idx)] for idx in topk_indices_cpu}
        return run, markdown_by_id


class ColEmbedSingletonRetriever:
    """
    A module-level singleton facade for ColEmbed retrieval.

    This wrapper provides explicit lifecycle calls (init/unload) while still
    maintaining a single global instance and hiding all retrieval complexity.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[_ColEmbedState] = None

    def init(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
        model_id: str = "nvidia/llama-nemoretriever-colembed-1b-v1",
        device: str = "cuda",
        top_k: int = 100,
        batch_size: int = 32,
        corpus_batch_size: int = 32,
        max_scoring_batch_size: int = 256,
        cache_dir: str | Path = "cache",
    ) -> None:
        """
        Initialize (or re-initialize) the singleton for a given dataset/corpus.

        - Model is loaded once per process (unless model_id/device changes).
        - Corpus embeddings are loaded from cache or computed once per dataset.
        """
        with self._lock:
            cache_dir = Path(cache_dir)

            # If state exists but model configuration changed, fully unload.
            if self._state is not None and (self._state.model_id != model_id or self._state.device != device):
                self.unload()

            if self._state is None:
                self._state = _ColEmbedState(
                    model_id=model_id,
                    device=device,
                    max_scoring_batch_size=max_scoring_batch_size,
                    batch_size=batch_size,
                    corpus_batch_size=corpus_batch_size,
                    top_k=top_k,
                    cache_dir=cache_dir,
                )
            else:
                # Update tunables (safe).
                self._state.top_k = top_k
                self._state.batch_size = batch_size
                self._state.corpus_batch_size = corpus_batch_size
                self._state.max_scoring_batch_size = max_scoring_batch_size
                self._state.cache_dir = cache_dir

            # If already initialized for the same dataset with same corpus_ids length, keep as-is.
            if (
                self._state.dataset_name == dataset_name
                and self._state.corpus_ids is not None
                and len(self._state.corpus_ids) == len(corpus_ids)
            ):
                return

            corpus_images = [doc["image"] for doc in corpus]
            corpus_markdown = [doc["markdown"] for doc in corpus]

            # (Re)load corpus embeddings for this dataset/corpus.
            corpus_embeddings_cpu = self._state._load_or_build_corpus_embeddings(
                dataset_name=dataset_name, corpus_ids=corpus_ids, corpus=corpus_images
            )

            self._state.dataset_name = dataset_name
            self._state.corpus_ids = list(corpus_ids)
            self._state.corpus_markdown = corpus_markdown
            self._state.corpus_embeddings_cpu = corpus_embeddings_cpu

            self._state.corpus_embeddings_gpu = None
            if corpus_embeddings_cpu.shape[0] <= self._state.max_scoring_batch_size:
                self._state.corpus_embeddings_gpu = corpus_embeddings_cpu.to(self._state.device)

    def retrieve(
        self, query: str, *, return_markdown: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        """
        Retrieve top-k corpus items for a single query.

        Note: This method intentionally uses a lock to remain safe if called from
        multiple threads in the future (GPU inference + shared model).
        """
        with self._lock:
            if self._state is None:
                raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")
            return self._state.retrieve_one(query, return_markdown=return_markdown)

    def unload(self) -> None:
        """Free model + embeddings and release GPU memory."""
        with self._lock:
            if self._state is None:
                return

            try:
                if self._state.corpus_embeddings_gpu is not None:
                    del self._state.corpus_embeddings_gpu
                if self._state.corpus_embeddings_cpu is not None:
                    del self._state.corpus_embeddings_cpu
                if self._state.corpus_markdown is not None:
                    del self._state.corpus_markdown
                if getattr(self._state, "model", None) is not None:
                    del self._state.model
            finally:
                self._state = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
retriever = ColEmbedSingletonRetriever()
