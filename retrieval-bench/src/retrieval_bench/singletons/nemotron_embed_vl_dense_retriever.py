# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Embed VL v2 singleton dense retriever (multimodal).

This module encapsulates heavy retrieval operations for:
  nvidia/llama-nemotron-embed-vl-1b-v2

Interface (module-level singleton):
  - init(...): load model + corpus embeddings (cached) once
  - retrieve(query): run retrieval for a single query string
  - unload(): free GPU/CPU memory

Design notes:
  - GPU-only by design (no CPU fallback); this model is too slow on CPU for our workflow.
  - Corpus embedding supports modality: image, text, image_text (default).
  - Query embedding is text-only via model.encode_queries().
  - Scores are cosine similarity via dot-product over L2-normalized embeddings.
  - Corpus embeddings are cached to disk (embeddings.pt + meta.json).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("Required dependencies not installed for Nemotron-VL dense retriever. Install: torch") from e

from retrieval_bench.singletons._shared import hash_corpus_ids10 as _hash_corpus_ids10
from retrieval_bench.singletons._shared import slugify as _slugify
from retrieval_bench.singletons._shared import try_preload_corpus_to_gpu as _try_preload_corpus_to_gpu


def _l2_normalize_fp32(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x32 = x.to(torch.float32)
    return x32 / (x32.norm(p=2, dim=-1, keepdim=True) + eps)


def _doc_len_by_modality(modality: str) -> int:
    m = str(modality or "").strip().lower()
    if m == "image":
        return 2048
    if m == "text":
        return 8192
    if m in ("image_text", "imagetext", "image+text"):
        return 10240
    raise ValueError(f"Unknown doc_modality '{modality}'. Expected: 'image', 'text', or 'image_text'.")


@dataclass(frozen=True, slots=True)
class _CacheMeta:
    dataset_name: str
    model_id: str
    doc_modality: str
    doc_max_length: int
    query_max_length: int
    max_input_tiles: int
    use_thumbnail: bool
    num_docs: int
    corpus_ids_hash10: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset_name": str(self.dataset_name),
            "model_id": str(self.model_id),
            "doc_modality": str(self.doc_modality),
            "doc_max_length": int(self.doc_max_length),
            "query_max_length": int(self.query_max_length),
            "max_input_tiles": int(self.max_input_tiles),
            "use_thumbnail": bool(self.use_thumbnail),
            "num_docs": int(self.num_docs),
            "corpus_ids_hash10": str(self.corpus_ids_hash10),
        }


class _NemotronVLDenseState:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        top_k: int,
        doc_modality: str,
        doc_max_length: int,
        query_max_length: int,
        corpus_batch_size: int,
        corpus_chunk_size: int,
        cache_dir: Path,
        max_input_tiles: int,
        use_thumbnail: bool,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.top_k = int(top_k)
        self.doc_modality = str(doc_modality)
        self.doc_max_length = int(doc_max_length)
        self.query_max_length = int(query_max_length)
        self.corpus_batch_size = int(corpus_batch_size)
        self.corpus_chunk_size = int(corpus_chunk_size)
        self.cache_dir = cache_dir
        self.max_input_tiles = int(max_input_tiles)
        self.use_thumbnail = bool(use_thumbnail)

        self.dataset_name: Optional[str] = None
        self.corpus_ids: Optional[List[str]] = None
        self.corpus_id_to_idx: Optional[Dict[str, int]] = None
        self.corpus_markdown: Optional[List[str]] = None
        self.corpus_embeddings_cpu: Optional[torch.Tensor] = None  # [n, dim] float16
        self.corpus_embeddings_gpu: Optional[torch.Tensor] = None  # [n, dim] float16

        self.model = self._load_model()
        self.processor = self._get_processor()
        self._fix_embedding_tokenizer_mismatch()
        self._configure_processor()

    def _load_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Nemotron-VL dense retriever is GPU-only.")

        # Compatibility shim for torch/transformers version skew (used elsewhere in repo).
        from retrieval_bench.utils.torch_compat import patch_torch_is_autocast_enabled

        patch_torch_is_autocast_enabled()

        try:
            from transformers import AutoModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("Missing dependency: transformers. Install it to use this retriever.") from e

        def _from_pretrained(*, attn_implementation: str):
            common_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": str(attn_implementation),
            }
            # Newer HF stacks deprecate `torch_dtype` in favor of `dtype`.
            try:
                return AutoModel.from_pretrained(
                    self.model_id,
                    dtype=torch.bfloat16,
                    **common_kwargs,
                )
            except TypeError:
                return AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    **common_kwargs,
                )

        # Prefer FlashAttention2 when available; fall back to eager.
        try:
            model = _from_pretrained(attn_implementation="flash_attention_2")
        except Exception:
            model = _from_pretrained(attn_implementation="eager")

        model.to("cuda")
        model.eval()
        return model

    def _get_processor(self):
        proc = getattr(self.model, "processor", None)
        if proc is None:
            raise RuntimeError(
                "Nemotron-VL model did not expose `model.processor`. "
                "This retriever expects a trust_remote_code model implementation with a processor attached."
            )
        return proc

    def _fix_embedding_tokenizer_mismatch(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return
        tok_vocab_size = len(tokenizer)
        emb = self.model.get_input_embeddings()
        if emb is None:
            return
        emb_rows = emb.weight.shape[0]
        if tok_vocab_size > emb_rows:
            dim = emb.weight.shape[1]
            new_weight = torch.zeros(tok_vocab_size, dim, dtype=emb.weight.dtype, device=emb.weight.device)
            new_weight[:emb_rows] = emb.weight.data
            emb.weight = torch.nn.Parameter(new_weight)
            emb.num_embeddings = tok_vocab_size

    def _configure_processor(self) -> None:
        # Tiling settings (model card defaults).
        if hasattr(self.processor, "max_input_tiles"):
            setattr(self.processor, "max_input_tiles", int(self.max_input_tiles))
        # Some implementations use `use_thumbnail`, others `use_thumbnails`.
        if hasattr(self.processor, "use_thumbnail"):
            setattr(self.processor, "use_thumbnail", bool(self.use_thumbnail))
        elif hasattr(self.processor, "use_thumbnails"):
            setattr(self.processor, "use_thumbnails", bool(self.use_thumbnail))

    def _set_processor_max_length_for_call(self, *, p_max_length: int) -> None:
        # The model card uses `processor.p_max_length` to control token budget.
        if hasattr(self.processor, "p_max_length"):
            setattr(self.processor, "p_max_length", int(p_max_length))

        # Best-effort: some processors also honor these.
        if hasattr(self.processor, "max_length"):
            setattr(self.processor, "max_length", int(p_max_length))

    def _index_dir(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        ds_slug = _slugify(dataset_name)
        model_slug = _slugify(self.model_id.split("/")[-1])
        mod_slug = _slugify(self.doc_modality)
        key = (
            f"{dataset_name}::{self.model_id}::{mod_slug}::"
            f"dlen{self.doc_max_length}::qlen{self.query_max_length}::"
            f"tiles{self.max_input_tiles}::thumb{int(self.use_thumbnail)}::{corpus_ids_hash10}"
        )
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / (
            f"nemotron_vl_dense__{ds_slug}__{model_slug}__{mod_slug}__dlen{self.doc_max_length}__"
            f"qlen{self.query_max_length}__tiles{self.max_input_tiles}__thumb{int(self.use_thumbnail)}__{key_hash}"
        )

    def _meta_path(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        return self._index_dir(dataset_name, corpus_ids_hash10=corpus_ids_hash10) / "meta.json"

    def _emb_path(self, dataset_name: str, *, corpus_ids_hash10: str) -> Path:
        return self._index_dir(dataset_name, corpus_ids_hash10=corpus_ids_hash10) / "embeddings.pt"

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
                doc_modality=str(data.get("doc_modality", "")),
                doc_max_length=int(data.get("doc_max_length", -1)),
                query_max_length=int(data.get("query_max_length", -1)),
                max_input_tiles=int(data.get("max_input_tiles", -1)),
                use_thumbnail=bool(data.get("use_thumbnail", False)),
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
            if str(meta.doc_modality) != str(self.doc_modality):
                return False
            if int(meta.doc_max_length) != int(self.doc_max_length):
                return False
            if int(meta.query_max_length) != int(self.query_max_length):
                return False
            if int(meta.max_input_tiles) != int(self.max_input_tiles):
                return False
            if bool(meta.use_thumbnail) != bool(self.use_thumbnail):
                return False
            if int(meta.num_docs) != int(num_docs):
                return False
            if str(meta.corpus_ids_hash10) != str(corpus_ids_hash10):
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

    def _embed_corpus_batched(self, corpus: Sequence[Dict[str, Any]]) -> torch.Tensor:
        bs = max(1, int(self.corpus_batch_size))
        out: List[torch.Tensor] = []
        modality = str(self.doc_modality).strip().lower()

        # Set doc max length for document calls.
        self._set_processor_max_length_for_call(p_max_length=int(self.doc_max_length))

        with torch.inference_mode():
            for i in range(0, len(corpus), bs):
                batch = corpus[i : i + bs]

                if modality == "image":
                    images = [doc["image"].convert("RGB") for doc in batch]
                    emb = self.model.encode_documents(images=images)
                elif modality == "text":
                    texts = [str(doc.get("markdown", "")) for doc in batch]
                    emb = self.model.encode_documents(texts=texts)
                else:  # image_text
                    images = [doc["image"].convert("RGB") for doc in batch]
                    texts = [str(doc.get("markdown", "")) for doc in batch]
                    emb = self.model.encode_documents(images=images, texts=texts)

                if not isinstance(emb, torch.Tensor):
                    raise RuntimeError(f"encode_documents returned unexpected type: {type(emb)}")
                if emb.ndim != 2:
                    raise RuntimeError(f"Unexpected document embedding shape: {tuple(emb.shape)}")

                emb = _l2_normalize_fp32(emb).to(torch.float16).detach().to("cpu")
                out.append(emb)

        return torch.cat(out, dim=0) if out else torch.empty((0, 0), dtype=torch.float16, device="cpu")

    def _load_or_build_corpus_embeddings(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        corpus_ids_list = [str(x) for x in corpus_ids]
        corpus_ids_hash10 = _hash_corpus_ids10(corpus_ids_list)

        emb_path = self._emb_path(dataset_name, corpus_ids_hash10=corpus_ids_hash10)
        meta = self._load_meta(dataset_name, corpus_ids_hash10=corpus_ids_hash10)

        if meta is None:
            print(f"[cache] dense rebuild: {emb_path} (missing)")
        elif not self._meta_matches(
            meta, dataset_name=dataset_name, corpus_ids_hash10=corpus_ids_hash10, num_docs=len(corpus_ids_list)
        ):
            print(f"[cache] dense rebuild: {emb_path} (meta_mismatch)")
        else:
            try:
                emb = torch.load(emb_path, map_location="cpu")
                if not isinstance(emb, torch.Tensor):
                    raise TypeError(f"Expected torch.Tensor in cache, got {type(emb)}")
                if emb.shape[0] != len(corpus_ids_list):
                    raise ValueError(
                        f"Cached embeddings mismatch: cached={emb.shape[0]} vs corpus={len(corpus_ids_list)}"
                    )
                print(f"[cache] dense hit: {emb_path}")
                return emb
            except Exception:
                print(f"[cache] dense rebuild: {emb_path} (load_error)")

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        emb = self._embed_corpus_batched(corpus)
        elapsed = time.time() - t0
        print(f"[cache] corpus embedding took {elapsed:.1f}s ({len(corpus)} docs)")

        tmp = emb_path.with_suffix(".pt.tmp")
        torch.save(emb, tmp)
        os.replace(tmp, emb_path)

        self._write_meta_atomic(
            _CacheMeta(
                dataset_name=str(dataset_name),
                model_id=str(self.model_id),
                doc_modality=str(self.doc_modality),
                doc_max_length=int(self.doc_max_length),
                query_max_length=int(self.query_max_length),
                max_input_tiles=int(self.max_input_tiles),
                use_thumbnail=bool(self.use_thumbnail),
                num_docs=int(len(corpus_ids_list)),
                corpus_ids_hash10=str(corpus_ids_hash10),
            ),
            dataset_name=dataset_name,
            corpus_ids_hash10=corpus_ids_hash10,
        )
        return emb

    def embed_query(self, query_text: str) -> torch.Tensor:
        # Set query max length for query call.
        self._set_processor_max_length_for_call(p_max_length=int(self.query_max_length))

        with torch.inference_mode():
            emb = self.model.encode_queries([str(query_text)])

        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(f"encode_queries returned unexpected type: {type(emb)}")
        if emb.ndim != 2 or emb.shape[0] != 1:
            raise RuntimeError(f"Unexpected query embedding shape: {tuple(emb.shape)}")

        emb1 = emb[0]
        emb1 = _l2_normalize_fp32(emb1).to(torch.float16).detach().to("cpu")
        return emb1  # [dim] on CPU

    def score_query(self, query_embedding_cpu: torch.Tensor) -> torch.Tensor:
        if self.corpus_embeddings_cpu is None:
            raise RuntimeError("corpus_embeddings_cpu is not set; call init() first")

        num_docs = self.corpus_embeddings_cpu.shape[0]
        scores_cpu = torch.empty((num_docs,), dtype=torch.float32, device="cpu")

        chunk = max(1, int(self.corpus_chunk_size))
        device = str(self.device)

        with torch.inference_mode():
            q_gpu = query_embedding_cpu.to(device, non_blocking=True)  # [dim]
            q_gpu = q_gpu.unsqueeze(1)  # [dim, 1]

            for c_start in range(0, num_docs, chunk):
                c_end = min(c_start + chunk, num_docs)
                if self.corpus_embeddings_gpu is not None:
                    c_gpu = self.corpus_embeddings_gpu[c_start:c_end]
                else:
                    c_gpu = self.corpus_embeddings_cpu[c_start:c_end].to(device, non_blocking=True)

                chunk_scores = torch.matmul(c_gpu, q_gpu).squeeze(1).float()  # [chunk]
                scores_cpu[c_start:c_end] = chunk_scores.to("cpu")

        return scores_cpu

    def retrieve_one(
        self,
        query: str,
        *,
        return_markdown: bool = False,
        excluded_ids: Optional[Sequence[str]] = None,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        if self.corpus_ids is None or self.corpus_embeddings_cpu is None:
            raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")

        q_emb_cpu = self.embed_query(str(query))
        scores_cpu = self.score_query(q_emb_cpu)

        # Apply per-query excluded ids BEFORE top-k selection (BRIGHT semantics).
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
                    pass

        k = min(int(self.top_k), len(self.corpus_ids))
        topk_scores, topk_indices = torch.topk(scores_cpu, k)
        ids = self.corpus_ids
        run = {ids[int(idx)]: float(score) for idx, score in zip(topk_indices.tolist(), topk_scores.tolist())}

        if not return_markdown:
            return run

        md = self.corpus_markdown or [""] * len(ids)
        markdown_by_id = {ids[int(idx)]: str(md[int(idx)]) for idx in topk_indices.tolist()}
        return run, markdown_by_id


class NemotronEmbedVLDenseSingletonRetriever:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[_NemotronVLDenseState] = None

    def init(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
        model_id: str = "nvidia/llama-nemotron-embed-vl-1b-v2",
        device: str = "auto",
        top_k: int = 100,
        doc_modality: str = "image_text",
        doc_max_length: Union[int, str] = "auto",
        query_max_length: int = 10240,
        corpus_batch_size: int = 4,
        corpus_chunk_size: int = 4096,
        cache_dir: str | Path = "cache/nemotron_vl_dense",
        preload_corpus_to_gpu: bool = False,
        max_input_tiles: int = 6,
        use_thumbnail: bool = True,
    ) -> None:
        with self._lock:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Nemotron-VL dense retriever is GPU-only.")

            device_eff = str(device or "auto").strip().lower()
            if device_eff in ("auto",):
                device_eff = "cuda"
            if not device_eff.startswith("cuda"):
                raise RuntimeError(f"Invalid device '{device}'. This retriever is GPU-only; use 'cuda'/'cuda:0'.")

            modality_eff = str(doc_modality or "image_text").strip()
            if isinstance(doc_max_length, str) and doc_max_length.strip().lower() == "auto":
                doc_max_length_eff = _doc_len_by_modality(modality_eff)
            else:
                doc_max_length_eff = int(doc_max_length)  # may raise, intended

            cache_dir_p = Path(cache_dir)

            # If state exists but config changed, unload.
            if self._state is not None and (
                self._state.model_id != str(model_id)
                or self._state.device != str(device_eff)
                or self._state.doc_modality != str(modality_eff)
                or int(self._state.doc_max_length) != int(doc_max_length_eff)
                or int(self._state.query_max_length) != int(query_max_length)
                or int(self._state.max_input_tiles) != int(max_input_tiles)
                or bool(self._state.use_thumbnail) != bool(use_thumbnail)
            ):
                self.unload()

            if self._state is None:
                self._state = _NemotronVLDenseState(
                    model_id=str(model_id),
                    device=str(device_eff),
                    top_k=int(top_k),
                    doc_modality=str(modality_eff),
                    doc_max_length=int(doc_max_length_eff),
                    query_max_length=int(query_max_length),
                    corpus_batch_size=int(corpus_batch_size),
                    corpus_chunk_size=int(corpus_chunk_size),
                    cache_dir=cache_dir_p,
                    max_input_tiles=int(max_input_tiles),
                    use_thumbnail=bool(use_thumbnail),
                )
            else:
                # Update tunables.
                self._state.top_k = int(top_k)
                self._state.corpus_batch_size = int(corpus_batch_size)
                self._state.corpus_chunk_size = int(corpus_chunk_size)
                self._state.cache_dir = cache_dir_p

            corpus_ids_list = [str(x) for x in corpus_ids]
            corpus_ids_hash10 = _hash_corpus_ids10(corpus_ids_list)

            if (
                self._state.dataset_name == str(dataset_name)
                and self._state.corpus_ids is not None
                and _hash_corpus_ids10(self._state.corpus_ids) == corpus_ids_hash10
                and self._state.corpus_embeddings_cpu is not None
            ):
                # Already initialized for the same corpus; only adjust GPU preload.
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
                corpus=list(corpus),
            )

            self._state.dataset_name = str(dataset_name)
            self._state.corpus_ids = corpus_ids_list
            self._state.corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids_list)}
            self._state.corpus_markdown = [str(doc.get("markdown", "")) for doc in corpus]
            self._state.corpus_embeddings_cpu = emb_cpu

            self._state.corpus_embeddings_gpu = None
            if preload_corpus_to_gpu:
                self._state.corpus_embeddings_gpu = _try_preload_corpus_to_gpu(emb_cpu, self._state.device)

    def retrieve(
        self,
        query: str,
        *,
        return_markdown: bool = False,
        excluded_ids: Optional[Sequence[str]] = None,
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
                if getattr(self._state, "processor", None) is not None:
                    del self._state.processor
            finally:
                self._state = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
retriever = NemotronEmbedVLDenseSingletonRetriever()
