# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HuggingFace dense-text singleton retriever.

This retriever is designed for instruction-tuned embedding models like:
  - hanhainebula/reason-embed-llama-3.1-8b-0928

Workflow assumptions (intentional):
- GPU-only: this model is too large for practical CPU use in our workflow.
- Corpus documents are embedded from the ViDoRe v3 `markdown` field (text).
- Queries default to the wrapper format:
    Instruct: <task_description>\nQuery: <query_text>
  but can alternatively use a full `query_prefix` (for models trained that way).
- Pooling is configurable: mean (default) or last-token pooling.

The exported `retriever` object is a module-level singleton with explicit lifecycle:
  init(...) -> retrieve(...) -> unload()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
except ImportError as e:  # pragma: no cover
    raise ImportError("Required dependencies not installed for HF dense retriever. Install: torch") from e

from retrieval_bench.singletons._shared import hash_corpus_ids10 as _hash_corpus_ids10
from retrieval_bench.singletons._shared import slugify as _slugify
from retrieval_bench.singletons._shared import try_preload_corpus_to_gpu as _try_preload_corpus_to_gpu


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool sentence embedding using the last non-pad token (handles left vs right padding).

    Copied from the reason-embed model card reference implementation.
    """
    # attention_mask: [bs, seq]
    # last_hidden_states: [bs, seq, dim]
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if bool(left_padding):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _mean_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool over non-pad tokens.

    attention_mask: [bs, seq] (0/1)
    last_hidden_states: [bs, seq, dim]
    """
    # Upcast is important for numeric stability (matches training/eval reference).
    last_hidden_states = last_hidden_states.to(torch.float32)
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1)[..., None].to(torch.float32).clamp(min=1.0)
    embedding = last_hidden_states_masked.sum(dim=1) / denom
    embedding = F.normalize(embedding, dim=-1)

    # Keep stored embeddings compact (CPU cache is large); compute in fp32, store in fp16.
    return embedding.to(torch.float16)


def _is_pathlike_model_id(model_id: str) -> bool:
    m = str(model_id or "")
    if not m:
        return False
    if m.startswith(("~", "/", "./", "../")):
        return True
    # If the expanded path exists locally, treat it as a path.
    try:
        p = Path(os.path.expanduser(m))
        return p.exists()
    except Exception:
        return False


def _normalize_model_id(model_id: str) -> str:
    m = str(model_id or "")
    if _is_pathlike_model_id(m):
        return os.path.expanduser(m)
    return m


def _wrap_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


@dataclass(frozen=True, slots=True)
class _CacheMeta:
    dataset_name: str
    model_id: str
    max_length: int
    pooling: str
    doc_prefix: str
    num_docs: int
    corpus_ids_hash10: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_id": self.model_id,
            "max_length": int(self.max_length),
            "pooling": str(self.pooling),
            "doc_prefix": str(self.doc_prefix),
            "num_docs": int(self.num_docs),
            "corpus_ids_hash10": self.corpus_ids_hash10,
        }


class _HfDenseState:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_length: int,
        pooling: str,
        doc_prefix: str,
        query_prefix: Optional[str],
        task_description: str,
        score_scale: float,
        batch_size: int,
        corpus_batch_size: int,
        scoring_batch_size: int,
        top_k: int,
        cache_dir: Path,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.max_length = int(max_length)
        self.pooling = str(pooling)
        self.doc_prefix = str(doc_prefix)
        self.query_prefix = str(query_prefix) if isinstance(query_prefix, str) else None
        self.task_description = str(task_description)
        self.score_scale = float(score_scale)
        self.batch_size = int(batch_size)
        self.corpus_batch_size = int(corpus_batch_size)
        self.scoring_batch_size = int(scoring_batch_size)
        self.top_k = int(top_k)
        self.cache_dir = cache_dir

        self.dataset_name: Optional[str] = None
        self.corpus_ids: Optional[List[str]] = None
        self.corpus_id_to_idx: Optional[Dict[str, int]] = None
        self.corpus_markdown: Optional[List[str]] = None
        self.corpus_embeddings_cpu: Optional[torch.Tensor] = None  # [n, dim] float16
        self.corpus_embeddings_gpu: Optional[torch.Tensor] = None  # [n, dim] float16

        self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mode = str(self.pooling or "last_token").strip().lower()
        if mode in ("mean", "avg", "average"):
            return _mean_pool(last_hidden_states, attention_mask)
        # Default / legacy.
        return _last_token_pool(last_hidden_states, attention_mask)

    def _load_model_and_tokenizer(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This dense retriever requires an NVIDIA GPU.")
        if not str(self.device).startswith("cuda"):
            raise RuntimeError(
                f"Invalid device '{self.device}'. This dense retriever is GPU-only; use 'cuda'/'cuda:0'."
            )

        # Compatibility shim for torch/transformers version skew.
        from retrieval_bench.utils.torch_compat import patch_torch_is_autocast_enabled

        patch_torch_is_autocast_enabled()

        from transformers import AutoModel, AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        model.eval()
        model.to(self.device)
        model.half()
        return tokenizer, model

    def _index_dir(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        ds_slug = _slugify(dataset_name)
        model_slug = _slugify(self.model_id.split("/")[-1])
        doc_slug = _slugify(self.doc_prefix)[:32]
        pool_slug = _slugify(self.pooling)
        key = f"{dataset_name}::{self.model_id}::{self.max_length}::{pool_slug}::{self.doc_prefix}::{corpus_ids_hash10}"
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        return (
            self.cache_dir
            / f"hf_dense__{ds_slug}__{model_slug}__len{self.max_length}__{pool_slug}__doc{doc_slug}__{key_hash}"
        )

    def _meta_path(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        return self._index_dir(dataset_name, corpus_ids_hash10=corpus_ids_hash10) / "meta.json"

    def _emb_path(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        return self._index_dir(dataset_name, corpus_ids_hash10=corpus_ids_hash10) / "embeddings.pt"

    def _build_meta(self, *, dataset_name: str, corpus_ids_hash10: str, num_docs: int) -> _CacheMeta:
        return _CacheMeta(
            dataset_name=str(dataset_name),
            model_id=str(self.model_id),
            max_length=int(self.max_length),
            pooling=str(self.pooling),
            doc_prefix=str(self.doc_prefix),
            num_docs=int(num_docs),
            corpus_ids_hash10=str(corpus_ids_hash10),
        )

    def _load_meta(self, dataset_name: str, *, corpus_ids_hash10: str) -> Optional[_CacheMeta]:
        p = self._meta_path(dataset_name, corpus_ids_hash10=corpus_ids_hash10)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return _CacheMeta(
                dataset_name=str(data.get("dataset_name", "")),
                model_id=str(data.get("model_id", "")),
                max_length=int(data.get("max_length", -1)),
                pooling=str(data.get("pooling", "")),
                doc_prefix=str(data.get("doc_prefix", "")),
                num_docs=int(data.get("num_docs", -1)),
                corpus_ids_hash10=str(data.get("corpus_ids_hash10", "")),
            )
        except Exception:
            return None

    def _meta_matches(self, meta: _CacheMeta, *, dataset_name: str, corpus_ids_hash10: str, num_docs: int) -> bool:
        try:
            if meta.dataset_name != str(dataset_name):
                return False
            if meta.model_id != str(self.model_id):
                return False
            if int(meta.max_length) != int(self.max_length):
                return False
            if str(meta.pooling) != str(self.pooling):
                return False
            if str(meta.doc_prefix) != str(self.doc_prefix):
                return False
            if int(meta.num_docs) != int(num_docs):
                return False
            if meta.corpus_ids_hash10 != str(corpus_ids_hash10):
                return False
            return True
        except Exception:
            return False

    def _write_meta_atomic(self, meta: _CacheMeta, *, dataset_name: str, corpus_ids_hash10: str) -> None:
        p = self._meta_path(dataset_name, corpus_ids_hash10=corpus_ids_hash10)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    def _tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts),
            max_length=int(self.max_length),
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def _embed_texts_batched(self, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
        out: List[torch.Tensor] = []
        bs = max(1, int(batch_size))

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                batch = self._tokenize(chunk)
                outputs = self.model(**batch)
                pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
                # Mean/avg pooling path already normalizes in fp32 and returns fp16.
                mode = str(self.pooling or "last_token").strip().lower()
                if mode not in ("mean", "avg", "average"):
                    pooled = F.normalize(pooled, p=2, dim=1)
                out.append(pooled.detach().to("cpu"))
        return torch.cat(out, dim=0) if out else torch.empty((0, 0), dtype=torch.float16, device="cpu")

    def _load_or_build_corpus_embeddings(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus_markdown: Sequence[str],
    ) -> torch.Tensor:
        corpus_ids_hash10 = _hash_corpus_ids10(corpus_ids)
        emb_path = self._emb_path(dataset_name, corpus_ids_hash10=corpus_ids_hash10)
        meta = self._load_meta(dataset_name, corpus_ids_hash10=corpus_ids_hash10)

        if meta is not None and self._meta_matches(
            meta, dataset_name=dataset_name, corpus_ids_hash10=corpus_ids_hash10, num_docs=len(corpus_ids)
        ):
            try:
                emb = torch.load(emb_path, map_location="cpu")
                if not isinstance(emb, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor in cache, got {type(emb)}")
                if emb.shape[0] != len(corpus_ids):
                    raise ValueError(f"Cached embeddings mismatch: cached={emb.shape[0]} vs corpus={len(corpus_ids)}")
                return emb
            except Exception:
                logger.debug("Cache load failed for %s, recomputing", emb_path, exc_info=True)

        # Build from scratch.
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        emb = self._embed_texts_batched(corpus_markdown, batch_size=int(self.corpus_batch_size))
        elapsed = time.time() - t0
        print(f"[cache] corpus embedding took {elapsed:.1f}s ({len(corpus_markdown)} docs)")

        tmp = emb_path.with_suffix(".pt.tmp")
        torch.save(emb, tmp)
        os.replace(tmp, emb_path)
        self._write_meta_atomic(
            self._build_meta(dataset_name=dataset_name, corpus_ids_hash10=corpus_ids_hash10, num_docs=len(corpus_ids)),
            dataset_name=dataset_name,
            corpus_ids_hash10=corpus_ids_hash10,
        )
        return emb

    def embed_query(self, query_text: str) -> torch.Tensor:
        if isinstance(self.query_prefix, str) and self.query_prefix:
            q = str(self.query_prefix) + str(query_text)
        else:
            q = _wrap_instruct(self.task_description, str(query_text))
        emb = self._embed_texts_batched([q], batch_size=1)
        if emb.ndim != 2 or emb.shape[0] != 1:
            raise RuntimeError(f"Unexpected query embedding shape: {tuple(emb.shape)}")
        return emb[0]  # [dim] on CPU

    def score_query(self, query_embedding_cpu: torch.Tensor) -> torch.Tensor:
        if self.corpus_embeddings_cpu is None:
            raise RuntimeError("corpus_embeddings_cpu is not set; call init() first")
        num_docs = self.corpus_embeddings_cpu.shape[0]
        scores_cpu = torch.empty((num_docs,), dtype=torch.float32, device="cpu")

        chunk = max(1, int(self.scoring_batch_size))
        scale = float(self.score_scale)

        with torch.no_grad():
            q_gpu = query_embedding_cpu.to(self.device, non_blocking=True)  # [dim]
            q_gpu = q_gpu.unsqueeze(1)  # [dim, 1]

            for c_start in range(0, num_docs, chunk):
                c_end = min(c_start + chunk, num_docs)
                if self.corpus_embeddings_gpu is not None:
                    c_gpu = self.corpus_embeddings_gpu[c_start:c_end]
                else:
                    c_gpu = self.corpus_embeddings_cpu[c_start:c_end].to(self.device, non_blocking=True)

                # [chunk, dim] @ [dim, 1] -> [chunk, 1]
                chunk_scores = torch.matmul(c_gpu, q_gpu).squeeze(1).float() * scale
                scores_cpu[c_start:c_end] = chunk_scores.to("cpu")

        return scores_cpu

    def retrieve_one(
        self,
        query: str,
        *,
        return_markdown: bool = False,
        excluded_ids: Optional[Sequence[str]] = None,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        if self.corpus_ids is None or self.corpus_embeddings_cpu is None or self.corpus_markdown is None:
            raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")

        q_emb_cpu = self.embed_query(query)
        scores_cpu = self.score_query(q_emb_cpu)

        # Apply per-query excluded ids BEFORE top-k selection (BRIGHT semantics).
        # This prevents excluded docs from "stealing" slots in top-k.
        if excluded_ids and self.corpus_id_to_idx:
            for did in set(str(x) for x in excluded_ids):
                if did == "N/A":
                    continue
                idx = self.corpus_id_to_idx.get(did, None)
                if idx is None:
                    continue
                try:
                    scores_cpu[int(idx)] = float("-inf")
                except Exception:
                    # Ignore malformed indices; keep scoring robust.
                    pass

        k = min(int(self.top_k), len(self.corpus_ids))
        topk_scores, topk_indices = torch.topk(scores_cpu, k)

        ids = self.corpus_ids
        run = {ids[int(idx)]: float(score) for idx, score in zip(topk_indices.tolist(), topk_scores.tolist())}

        if not return_markdown:
            return run

        md = self.corpus_markdown
        markdown_by_id = {ids[int(idx)]: md[int(idx)] for idx in topk_indices.tolist()}
        return run, markdown_by_id


class HfDenseSingletonRetriever:
    """
    Module-level singleton facade.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[_HfDenseState] = None

    def init(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
        model_id: str,
        device: str = "cuda",
        top_k: int = 100,
        max_length: int = 8192,
        pooling: str = "mean",
        doc_prefix: str = "",
        query_prefix: Optional[str] = None,
        task_description: str = "Given the following post, retrieve relevant passages that help answer the post.",
        score_scale: float = 100.0,
        batch_size: int = 1,
        corpus_batch_size: int = 1,
        scoring_batch_size: int = 4096,
        cache_dir: str | Path = "cache/hf_dense",
        preload_corpus_to_gpu: bool = False,
    ) -> None:
        """
        Initialize (or re-initialize) the singleton for a given dataset/corpus.
        """
        with self._lock:
            cache_dir = Path(cache_dir)
            model_id_norm = _normalize_model_id(model_id)

            if self._state is not None and (
                self._state.model_id != str(model_id_norm)
                or self._state.device != str(device)
                or int(self._state.max_length) != int(max_length)
                or str(self._state.pooling) != str(pooling)
                or str(self._state.doc_prefix) != str(doc_prefix)
            ):
                self.unload()

            if self._state is None:
                self._state = _HfDenseState(
                    model_id=str(model_id_norm),
                    device=str(device),
                    max_length=int(max_length),
                    pooling=str(pooling),
                    doc_prefix=str(doc_prefix),
                    query_prefix=query_prefix,
                    task_description=str(task_description),
                    score_scale=float(score_scale),
                    batch_size=int(batch_size),
                    corpus_batch_size=int(corpus_batch_size),
                    scoring_batch_size=int(scoring_batch_size),
                    top_k=int(top_k),
                    cache_dir=cache_dir,
                )
            else:
                # Update tunables.
                self._state.top_k = int(top_k)
                self._state.batch_size = int(batch_size)
                self._state.corpus_batch_size = int(corpus_batch_size)
                self._state.scoring_batch_size = int(scoring_batch_size)
                self._state.cache_dir = cache_dir
                self._state.task_description = str(task_description)
                self._state.query_prefix = str(query_prefix) if isinstance(query_prefix, str) else None
                self._state.score_scale = float(score_scale)

            corpus_markdown = [str(doc.get("markdown", "")) for doc in corpus]
            corpus_ids_list = [str(x) for x in corpus_ids]
            corpus_texts_for_embed = [str(self._state.doc_prefix) + md for md in corpus_markdown]

            corpus_ids_hash10 = _hash_corpus_ids10(corpus_ids_list)
            if (
                self._state.dataset_name == str(dataset_name)
                and self._state.corpus_ids is not None
                and _hash_corpus_ids10(self._state.corpus_ids) == corpus_ids_hash10
                and self._state.corpus_embeddings_cpu is not None
            ):
                # Already initialized for the same corpus; only (possibly) update GPU preload.
                if preload_corpus_to_gpu and self._state.corpus_embeddings_gpu is None:
                    self._state.corpus_embeddings_gpu = _try_preload_corpus_to_gpu(
                        self._state.corpus_embeddings_cpu, self._state.device
                    )
                if (not preload_corpus_to_gpu) and self._state.corpus_embeddings_gpu is not None:
                    self._state.corpus_embeddings_gpu = None
                return

            emb_cpu = self._state._load_or_build_corpus_embeddings(
                dataset_name=str(dataset_name),
                corpus_ids=corpus_ids_list,
                corpus_markdown=corpus_texts_for_embed,
            )

            self._state.dataset_name = str(dataset_name)
            self._state.corpus_ids = corpus_ids_list
            self._state.corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids_list)}
            self._state.corpus_markdown = corpus_markdown
            self._state.corpus_embeddings_cpu = emb_cpu

            self._state.corpus_embeddings_gpu = None
            if preload_corpus_to_gpu:
                self._state.corpus_embeddings_gpu = _try_preload_corpus_to_gpu(emb_cpu, self._state.device)

    def retrieve(
        self, query: str, *, return_markdown: bool = False, excluded_ids: Optional[Sequence[str]] = None
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        with self._lock:
            if self._state is None:
                raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")
            return self._state.retrieve_one(
                str(query),
                return_markdown=bool(return_markdown),
                excluded_ids=excluded_ids,
            )

    def unload(self) -> None:
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
                if getattr(self._state, "tokenizer", None) is not None:
                    del self._state.tokenizer
            finally:
                self._state = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
retriever = HfDenseSingletonRetriever()
