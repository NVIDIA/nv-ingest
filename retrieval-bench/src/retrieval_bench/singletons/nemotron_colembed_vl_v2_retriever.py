# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron ColEmbed-VL v2 singleton retriever (late interaction, ColBERT-style MaxSim).

Backed model:
  - nvidia/nemotron-colembed-vl-8b-v2

Workflow:
  - Corpus documents are embedded from the ViDoRe `image` field (PIL images).
  - Queries are embedded from text.
  - Scoring is explicit ColBERT MaxSim (matmul + max over doc tokens + sum over query tokens),
    computed on GPU in bounded corpus chunks.

Interface (module-level singleton):
  - init(...): load model + load/build cached corpus embeddings once per dataset/corpus
  - retrieve(query): retrieve top-k for a single query; optionally return markdown for agentic prompts
  - unload(): free model + embeddings and release GPU memory
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Required dependencies not installed for Nemotron ColEmbed-VL v2 retriever. "
        "Please install at least: torch (and for actual retrieval: transformers, optionally flash-attn)."
    ) from e


def _set_tiling_knobs_if_present(model: Any, *, max_input_tiles: int, use_thumbnail: bool) -> None:
    """
    Best-effort configuration of the remote-code processor tiling knobs.

    The model card recommends:
      - max_input_tiles = 8
      - use_thumbnails = True
    """

    proc = getattr(model, "processor", None)
    if proc is None:
        return

    if hasattr(proc, "max_input_tiles"):
        try:
            setattr(proc, "max_input_tiles", int(max_input_tiles))
        except Exception:
            pass

    # Some implementations call it `use_thumbnail`, some `use_thumbnails`.
    if hasattr(proc, "use_thumbnail"):
        try:
            setattr(proc, "use_thumbnail", bool(use_thumbnail))
        except Exception:
            pass
    elif hasattr(proc, "use_thumbnails"):
        try:
            setattr(proc, "use_thumbnails", bool(use_thumbnail))
        except Exception:
            pass


def _balanced_chunk_size(num_items: int, max_chunk: int) -> int:
    if num_items <= max_chunk:
        return num_items
    k = math.ceil(num_items / max_chunk)
    return math.ceil(num_items / k)


class _NemotronColEmbedVLV2State:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_scoring_batch_size: int,
        scoring_chunk_size: int,
        corpus_batch_size: int,
        top_k: int,
        cache_dir: Path,
        max_input_tiles: int,
        use_thumbnail: bool,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.max_scoring_batch_size = int(max_scoring_batch_size)
        self.scoring_chunk_size = int(scoring_chunk_size)
        self.corpus_batch_size = int(corpus_batch_size)
        self.top_k = int(top_k)
        self.cache_dir = cache_dir
        self.max_input_tiles = int(max_input_tiles)
        self.use_thumbnail = bool(use_thumbnail)

        self.dataset_name: Optional[str] = None
        self.corpus_ids: Optional[List[str]] = None
        self.corpus_id_to_idx: Optional[Dict[str, int]] = None
        self.corpus_markdown: Optional[List[str]] = None
        self.corpus_embeddings_cpu: Optional[torch.Tensor] = None
        self.corpus_embeddings_gpu: Optional[torch.Tensor] = None
        self.corpus_token_lengths_cpu: Optional[torch.Tensor] = None
        self.corpus_token_lengths_gpu: Optional[torch.Tensor] = None

        self.model = self._load_model()
        _set_tiling_knobs_if_present(
            self.model,
            max_input_tiles=int(self.max_input_tiles),
            use_thumbnail=bool(self.use_thumbnail),
        )

    def _load_model(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Nemotron ColEmbed-VL v2 retriever requires an NVIDIA GPU.")

        # Compatibility shim for torch/transformers version skew.
        from retrieval_bench.utils.torch_compat import patch_torch_is_autocast_enabled

        patch_torch_is_autocast_enabled()

        # Lazy import so importing this module doesn't require transformers.
        from transformers import AutoModel  # type: ignore

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

    def _corpus_cache_path(self, dataset_name: str) -> Path:
        dataset_slug = str(dataset_name).replace("/", "__")
        model_slug = self.model_id.split("/")[-1].replace("/", "__")
        key = f"{dataset_name}::{self.model_id}::images::max_input_tiles={int(self.max_input_tiles)}::use_thumbnail={bool(self.use_thumbnail)}"
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        filename = f"corpus_image_embeddings__{dataset_slug}__{model_slug}__{key_hash}.pt"
        return self.cache_dir / filename

    def _embed_corpus_batched(self, corpus_images: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not corpus_images:
            return (
                torch.empty((0, 0, 0), dtype=torch.float32, device="cpu"),
                torch.empty((0,), dtype=torch.int32, device="cpu"),
            )

        imgs = [img.convert("RGB") for img in corpus_images]
        bs = max(1, int(self.corpus_batch_size))

        with torch.inference_mode():
            emb = self.model.forward_images(imgs, batch_size=bs)

        if not isinstance(emb, torch.Tensor):
            raise RuntimeError(f"forward_images returned unexpected type: {type(emb)}")

        emb = emb.to("cpu")

        try:
            token_norms = emb.float().norm(dim=-1)
            lengths = (token_norms > 1e-6).sum(dim=1).to(dtype=torch.int32)
        except Exception:
            lengths = torch.full((emb.shape[0],), emb.shape[1], dtype=torch.int32)

        return emb, lengths

    def _load_or_build_corpus_embeddings(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus_images: Sequence[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._corpus_cache_path(dataset_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not cache_path.exists():
            print(f"[cache] dense rebuild: {cache_path} (missing)")
        else:
            try:
                obj = torch.load(cache_path, map_location="cpu")
                if isinstance(obj, torch.Tensor):
                    emb = obj
                    lengths = torch.full((int(emb.shape[0]),), int(emb.shape[1]), dtype=torch.int32, device="cpu")
                elif isinstance(obj, dict) and isinstance(obj.get("embeddings", None), torch.Tensor):
                    emb = obj["embeddings"]
                    lengths_obj = obj.get("lengths", None)
                    if isinstance(lengths_obj, torch.Tensor):
                        lengths = lengths_obj.to("cpu", dtype=torch.int32)
                    else:
                        lengths = torch.full((int(emb.shape[0]),), int(emb.shape[1]), dtype=torch.int32, device="cpu")
                else:
                    raise TypeError(f"Expected torch.Tensor or dict cache, got {type(obj)}")

                if int(emb.shape[0]) != int(len(corpus_ids)):
                    raise ValueError(
                        f"Cached embeddings mismatch: cached={emb.shape[0]} vs corpus_ids={len(corpus_ids)}"
                    )
                if int(lengths.shape[0]) != int(emb.shape[0]):
                    lengths = torch.full((int(emb.shape[0]),), int(emb.shape[1]), dtype=torch.int32, device="cpu")
                lengths = torch.clamp(lengths.to("cpu", dtype=torch.int32), min=0, max=int(emb.shape[1]))
                print(f"[cache] dense hit: {cache_path}")
                return emb, lengths
            except Exception:
                # fall through to recompute
                print(f"[cache] dense rebuild: {cache_path} (load_error)")

        t0 = time.time()
        emb, lengths = self._embed_corpus_batched(corpus_images)
        elapsed = time.time() - t0
        print(f"[cache] corpus embedding took {elapsed:.1f}s ({len(corpus_images)} docs)")
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save({"embeddings": emb, "lengths": lengths}, tmp_path)
        os.replace(tmp_path, cache_path)
        return emb, lengths

    def _embed_query(self, query: str) -> torch.Tensor:
        with torch.no_grad():
            q_emb = self.model.forward_queries([str(query)], batch_size=1).detach()
        if not isinstance(q_emb, torch.Tensor) or q_emb.ndim != 3 or q_emb.shape[0] != 1:
            raise RuntimeError(f"Unexpected query embedding shape: {getattr(q_emb, 'shape', None)}")
        return q_emb[0].to(self.device)

    def _score_maxsim_block(
        self,
        emb_block: torch.Tensor,
        len_block: torch.Tensor,
        q_t: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        token_sims = torch.matmul(emb_block, q_t)  # [block, c_seq, q_seq]
        c_seq = int(token_sims.shape[1])
        pos = torch.arange(c_seq, device=device).unsqueeze(0)
        valid = pos < len_block.unsqueeze(1)
        token_sims = token_sims.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        return token_sims.max(dim=1).values.float().sum(dim=1)

    def _score_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        emb_gpu = self.corpus_embeddings_gpu
        emb_cpu = self.corpus_embeddings_cpu
        len_gpu = self.corpus_token_lengths_gpu
        len_cpu = self.corpus_token_lengths_cpu
        if emb_gpu is None and emb_cpu is None:
            raise RuntimeError("No corpus embeddings available.")
        if len_gpu is None and len_cpu is None:
            raise RuntimeError("No corpus token lengths available.")

        source = emb_gpu if emb_gpu is not None else emb_cpu
        num_corpus = source.shape[0]
        device = str(self.device)
        scores = torch.empty((num_corpus,), dtype=torch.float32, device=device)

        with torch.no_grad():
            q_t = query_embedding.transpose(0, 1)  # [dim, q_seq]

            if emb_gpu is not None:
                scores[:] = self._score_maxsim_block(emb_gpu, len_gpu, q_t, device)
            else:
                transfer_chunk = _balanced_chunk_size(num_corpus, max(1, int(self.max_scoring_batch_size)))
                score_chunk = max(1, int(self.scoring_chunk_size))

                for t_start in range(0, num_corpus, transfer_chunk):
                    t_end = min(t_start + transfer_chunk, num_corpus)
                    t_emb = emb_cpu[t_start:t_end].to(device)
                    t_len = len_cpu[t_start:t_end].to(device)

                    for s_start in range(0, t_emb.shape[0], score_chunk):
                        s_end = min(s_start + score_chunk, t_emb.shape[0])
                        block_scores = self._score_maxsim_block(
                            t_emb[s_start:s_end],
                            t_len[s_start:s_end],
                            q_t,
                            device,
                        )
                        scores[t_start + s_start : t_start + s_end] = block_scores

                    del t_emb, t_len

        return scores

    def retrieve_one(
        self,
        query: str,
        *,
        return_markdown: bool = False,
        excluded_ids: Optional[Sequence[str]] = None,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        if self.corpus_ids is None or (self.corpus_embeddings_gpu is None and self.corpus_embeddings_cpu is None):
            raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")

        q_emb = self._embed_query(str(query))
        scores = self._score_query(q_emb)

        if excluded_ids and self.corpus_id_to_idx:
            excluded_indices = []
            for did in set(str(x) for x in excluded_ids):
                if did == "N/A":
                    continue
                idx = self.corpus_id_to_idx.get(did, None)
                if idx is not None:
                    excluded_indices.append(int(idx))
            if excluded_indices:
                scores[torch.tensor(excluded_indices, device=scores.device)] = float("-inf")

        k = min(int(self.top_k), len(self.corpus_ids))
        topk_scores, topk_indices = torch.topk(scores, k)
        ids = self.corpus_ids
        topk_indices_cpu = topk_indices.cpu().tolist()
        topk_scores_cpu = topk_scores.cpu().tolist()
        run = {ids[int(idx)]: float(score) for idx, score in zip(topk_indices_cpu, topk_scores_cpu)}

        if not return_markdown:
            return run

        md = self.corpus_markdown or [""] * len(ids)
        markdown_by_id = {ids[int(idx)]: str(md[int(idx)]) for idx in topk_indices_cpu}
        return run, markdown_by_id


class NemotronColEmbedVLV2SingletonRetriever:
    """
    Module-level singleton facade for Nemotron ColEmbed-VL v2 retrieval.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[_NemotronColEmbedVLV2State] = None

    def init(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
        model_id: str = "nvidia/nemotron-colembed-vl-8b-v2",
        device: str = "cuda",
        top_k: int = 100,
        corpus_batch_size: int = 32,
        max_scoring_batch_size: int = 3000,
        scoring_chunk_size: int = 1311,
        cache_dir: str | Path = "cache/nemotron_colembed_vl_v2",
        max_input_tiles: int = 8,
        use_thumbnail: bool = True,
    ) -> None:
        with self._lock:
            cache_dir_p = Path(cache_dir)

            # If state exists but model configuration changed, unload.
            if self._state is not None and (self._state.model_id != str(model_id) or self._state.device != str(device)):
                self.unload()

            if self._state is None:
                self._state = _NemotronColEmbedVLV2State(
                    model_id=str(model_id),
                    device=str(device),
                    max_scoring_batch_size=int(max_scoring_batch_size),
                    scoring_chunk_size=int(scoring_chunk_size),
                    corpus_batch_size=int(corpus_batch_size),
                    top_k=int(top_k),
                    cache_dir=cache_dir_p,
                    max_input_tiles=int(max_input_tiles),
                    use_thumbnail=bool(use_thumbnail),
                )
            else:
                # Update tunables.
                self._state.top_k = int(top_k)
                self._state.corpus_batch_size = int(corpus_batch_size)
                self._state.max_scoring_batch_size = int(max_scoring_batch_size)
                self._state.scoring_chunk_size = int(scoring_chunk_size)
                self._state.cache_dir = cache_dir_p
                self._state.max_input_tiles = int(max_input_tiles)
                self._state.use_thumbnail = bool(use_thumbnail)

            # If already initialized for same dataset and same corpus length, keep as-is (fast path).
            if (
                self._state.dataset_name == str(dataset_name)
                and self._state.corpus_ids is not None
                and len(self._state.corpus_ids) == len(corpus_ids)
                and self._state.corpus_embeddings_cpu is not None
                and self._state.corpus_token_lengths_cpu is not None
            ):
                should_be_on_gpu = len(corpus_ids) <= self._state.max_scoring_batch_size
                if should_be_on_gpu and self._state.corpus_embeddings_gpu is None:
                    self._state.corpus_embeddings_gpu = self._state.corpus_embeddings_cpu.to(self._state.device)
                if should_be_on_gpu and self._state.corpus_token_lengths_gpu is None:
                    self._state.corpus_token_lengths_gpu = self._state.corpus_token_lengths_cpu.to(self._state.device)
                if (not should_be_on_gpu) and self._state.corpus_embeddings_gpu is not None:
                    self._state.corpus_embeddings_gpu = None
                if (not should_be_on_gpu) and self._state.corpus_token_lengths_gpu is not None:
                    self._state.corpus_token_lengths_gpu = None
                return

            corpus_images = [doc["image"] for doc in corpus]
            corpus_markdown = [str(doc.get("markdown", "")) for doc in corpus]

            emb_cpu, lengths_cpu = self._state._load_or_build_corpus_embeddings(
                dataset_name=str(dataset_name),
                corpus_ids=corpus_ids,
                corpus_images=corpus_images,
            )

            self._state.dataset_name = str(dataset_name)
            corpus_ids_list = [str(x) for x in corpus_ids]
            self._state.corpus_ids = corpus_ids_list
            self._state.corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids_list)}
            self._state.corpus_markdown = corpus_markdown
            self._state.corpus_embeddings_cpu = emb_cpu
            self._state.corpus_token_lengths_cpu = lengths_cpu

            self._state.corpus_embeddings_gpu = None
            self._state.corpus_token_lengths_gpu = None
            if emb_cpu.shape[0] <= self._state.max_scoring_batch_size:
                self._state.corpus_embeddings_gpu = emb_cpu.to(self._state.device)
                self._state.corpus_token_lengths_gpu = lengths_cpu.to(self._state.device)

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
                if self._state.corpus_token_lengths_gpu is not None:
                    del self._state.corpus_token_lengths_gpu
                if self._state.corpus_embeddings_cpu is not None:
                    del self._state.corpus_embeddings_cpu
                if self._state.corpus_token_lengths_cpu is not None:
                    del self._state.corpus_token_lengths_cpu
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
retriever = NemotronColEmbedVLV2SingletonRetriever()
