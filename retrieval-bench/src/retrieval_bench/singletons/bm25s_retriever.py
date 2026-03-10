# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BM25S singleton retriever (keyword / lexical search).

This module encapsulates BM25 retrieval via `bm25s` behind a small interface:

- init(...): build/load a cached BM25 index once per dataset/corpus
- retrieve(query): run BM25 retrieval for a single query string
- unload(): free memory

Important: BM25S returns positional doc indices; we always map indices back to
the provided `corpus_ids` so the pipeline returns evaluator-compatible ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from retrieval_bench.singletons._shared import hash_corpus_ids10 as _hash_corpus_ids10
from retrieval_bench.utils.corpus_language import corpus_language


def _stemmer_language(dataset_name: str) -> str:
    # Workflow rule: stemmer is keyed off corpus language, which depends on dataset.
    # (Delegated to shared helper so other components can reuse this decision.)
    return corpus_language(dataset_name)


def _stopwords_language(stemmer_language: str) -> str:
    # bm25s uses short language codes for built-in stopwords.
    return "fr" if stemmer_language == "french" else "en"


@dataclass(frozen=True, slots=True)
class _CacheMeta:
    dataset_name: str
    stemmer_language: str
    num_docs: int
    corpus_ids_hash10: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "stemmer_language": self.stemmer_language,
            "num_docs": int(self.num_docs),
            "corpus_ids_hash10": self.corpus_ids_hash10,
        }


class _Bm25sState:
    def __init__(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus_markdown: Sequence[str],
        top_k: int,
        cache_dir: Path,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.top_k = int(top_k)
        self.cache_dir = cache_dir

        self.corpus_ids: List[str] = [str(x) for x in corpus_ids]
        self.corpus_markdown: List[str] = [str(x) for x in corpus_markdown]
        self.stemmer_language = _stemmer_language(self.dataset_name)
        self.stopwords_language = _stopwords_language(self.stemmer_language)
        self.corpus_ids_hash10 = self._corpus_ids_hash10()

        self._bm25 = None
        self._stemmer = None

        self._load_or_build_index()

    def _cache_key_hash10(self) -> str:
        key = f"{self.dataset_name}::{self.stemmer_language}::bm25s"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]

    def _corpus_ids_hash10(self) -> str:
        return _hash_corpus_ids10(self.corpus_ids)

    def _index_dir(self) -> Path:
        ds_slug = self.dataset_name.replace("/", "__")
        key_hash = self._cache_key_hash10()
        return self.cache_dir / f"bm25s_index__{ds_slug}__{self.stemmer_language}__{key_hash}"

    def _meta_path(self) -> Path:
        return self._index_dir() / "meta.json"

    def _build_meta(self) -> _CacheMeta:
        return _CacheMeta(
            dataset_name=self.dataset_name,
            stemmer_language=self.stemmer_language,
            num_docs=len(self.corpus_ids),
            corpus_ids_hash10=self.corpus_ids_hash10,
        )

    def _load_meta(self) -> Optional[_CacheMeta]:
        p = self._meta_path()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return _CacheMeta(
                dataset_name=str(data.get("dataset_name", "")),
                stemmer_language=str(data.get("stemmer_language", "")),
                num_docs=int(data.get("num_docs", -1)),
                corpus_ids_hash10=str(data.get("corpus_ids_hash10", "")),
            )
        except Exception:
            return None

    def _write_meta_atomic(self, meta: _CacheMeta) -> None:
        p = self._meta_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    def _meta_matches(self, meta: _CacheMeta) -> bool:
        try:
            if meta.dataset_name != self.dataset_name:
                return False
            if meta.stemmer_language != self.stemmer_language:
                return False
            if int(meta.num_docs) != len(self.corpus_ids):
                return False
            if meta.corpus_ids_hash10 != self.corpus_ids_hash10:
                return False
            return True
        except Exception:
            return False

    def _load_or_build_index(self) -> None:
        try:
            import bm25s  # type: ignore
            import Stemmer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Missing dependencies for BM25S singleton. Install with `pip install bm25s PyStemmer` "
                "(or ensure they are in project dependencies)."
            ) from e

        self._stemmer = Stemmer.Stemmer(self.stemmer_language)
        index_dir = self._index_dir()
        meta = self._load_meta()

        if meta is not None and self._meta_matches(meta):
            try:
                self._bm25 = bm25s.BM25.load(str(index_dir))
                return
            except Exception:
                # Fall through to rebuild.
                self._bm25 = None

        # Build from scratch and persist.
        corpus_tokens = bm25s.tokenize(
            self.corpus_markdown,
            stopwords=self.stopwords_language,
            stemmer=self._stemmer,
        )
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)

        index_dir.mkdir(parents=True, exist_ok=True)
        bm25.save(str(index_dir))
        self._write_meta_atomic(self._build_meta())
        self._bm25 = bm25

    def retrieve_one(
        self, query: str, *, return_markdown: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        if self._bm25 is None or self._stemmer is None:
            raise RuntimeError("BM25S retriever not initialized. Call retriever.init(...) first.")

        import bm25s  # type: ignore

        query_tokens = bm25s.tokenize(
            [str(query)],
            stopwords=self.stopwords_language,
            stemmer=self._stemmer,
        )

        k = min(int(self.top_k), len(self.corpus_ids))
        results_idx, scores = self._bm25.retrieve(query_tokens, k=k)

        # bm25s returns arrays of shape (n_queries, k); we pass a single query.
        idxs = results_idx[0]
        scs = scores[0]

        run: Dict[str, float] = {}
        markdown_by_id: Dict[str, str] = {}

        for doc_pos, score in zip(list(idxs), list(scs)):
            pos = int(doc_pos)
            if pos < 0 or pos >= len(self.corpus_ids):
                continue
            doc_id = self.corpus_ids[pos]
            run[doc_id] = float(score)
            if return_markdown:
                markdown_by_id[doc_id] = self.corpus_markdown[pos]

        if not return_markdown:
            return run
        return run, markdown_by_id


class Bm25sSingletonRetriever:
    """
    A module-level singleton facade for bm25s retrieval.

    This wrapper provides explicit lifecycle calls (init/unload) while still
    maintaining a single global instance and hiding indexing/caching complexity.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Optional[_Bm25sState] = None

    def init(
        self,
        *,
        dataset_name: str,
        corpus_ids: Sequence[str],
        corpus: Sequence[Dict[str, Any]],
        top_k: int = 100,
        cache_dir: str | Path = "cache/bm25s",
    ) -> None:
        with self._lock:
            cache_dir = Path(cache_dir)

            corpus_markdown = [str(doc.get("markdown", "")) for doc in corpus]

            # If already initialized for the same dataset and same corpus ids, keep as-is (fast path).
            if (
                self._state is not None
                and self._state.dataset_name == str(dataset_name)
                and self._state.corpus_ids_hash10 == _hash_corpus_ids10(corpus_ids)
            ):
                self._state.top_k = int(top_k)
                self._state.cache_dir = cache_dir
                return

            self._state = _Bm25sState(
                dataset_name=str(dataset_name),
                corpus_ids=corpus_ids,
                corpus_markdown=corpus_markdown,
                top_k=int(top_k),
                cache_dir=cache_dir,
            )

    def retrieve(
        self, query: str, *, return_markdown: bool = False
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        with self._lock:
            if self._state is None:
                raise RuntimeError("Retriever not initialized. Call retriever.init(...) first.")
            return self._state.retrieve_one(query, return_markdown=return_markdown)

    def unload(self) -> None:
        with self._lock:
            self._state = None


# ---------------------------------------------------------------------------
# Module-level singleton instance
# ---------------------------------------------------------------------------
retriever = Bm25sSingletonRetriever()
