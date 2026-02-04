from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx
import torch

from ..nim.http_utils import default_headers, normalize_endpoint


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x.float()
    denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / denom


def _coerce_embeddings_from_response(obj: object) -> List[List[float]]:
    """
    Best-effort parsing of common embeddings response shapes.

    Supports:
      - OpenAI-style: {"data": [{"embedding": [...]}, ...]}
      - Flat: {"embeddings": [[...], ...]} or {"embedding": [...]}
      - Direct list: [[...], ...]
    """
    if isinstance(obj, list):
        # Either [[...], ...] or [{"embedding": ...}, ...]
        if obj and isinstance(obj[0], dict) and "embedding" in obj[0]:
            return [list(map(float, x.get("embedding") or [])) for x in obj]  # type: ignore[arg-type]
        return [list(map(float, x)) for x in obj]  # type: ignore[arg-type]

    if not isinstance(obj, dict):
        raise ValueError(f"Unrecognized embeddings response type: {type(obj)}")

    if "data" in obj and isinstance(obj["data"], list):
        out: List[List[float]] = []
        for item in obj["data"]:
            if isinstance(item, dict) and "embedding" in item:
                out.append([float(v) for v in (item["embedding"] or [])])
        if out:
            return out

    if "embeddings" in obj and isinstance(obj["embeddings"], list):
        return [[float(v) for v in row] for row in obj["embeddings"]]

    if "embedding" in obj and isinstance(obj["embedding"], list):
        return [[float(v) for v in obj["embedding"]]]

    raise ValueError(f"Could not parse embeddings from response keys: {list(obj.keys())[:10]}")


@dataclass
class LlamaNemotronEmbed1BV2Embedder:
    """
    Minimal embedder wrapper that can run either:
      - locally via `llama_nemotron_embed_1b_v2` (GPU/CPU), or
      - remotely via an OpenAI-compatible embeddings endpoint.
    """

    endpoint: Optional[str] = None
    model_name: Optional[str] = None
    timeout_seconds: float = 60.0
    headers: Optional[Dict[str, str]] = None
    normalize: bool = True

    def __post_init__(self) -> None:
        self._endpoint = normalize_endpoint(self.endpoint) if self.endpoint else None
        self._headers = default_headers(self.headers)
        self._timeout_seconds = float(self.timeout_seconds)

        self._tokenizer = None
        self._model = None
        self._device = None

        if self._endpoint is None:
            import llama_nemotron_embed_1b_v2

            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hf_cache_dir = str(Path.home() / ".cache" / "huggingface")
            self._tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer(cache_dir=hf_cache_dir, force_download=False)
            self._model = llama_nemotron_embed_1b_v2.load_model(
                device=str(dev), trust_remote_code=True, cache_dir=hf_cache_dir, force_download=False
            )
            self._model.eval()
            self._device = dev

    @property
    def is_remote(self) -> bool:
        return self._endpoint is not None

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """
        Returns a CPU tensor of shape [N, D].
        """
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        if self._endpoint is not None:
            return self._embed_remote(texts_list, batch_size=batch_size)
        return self._embed_local(texts_list, batch_size=batch_size)

    def _embed_local(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local embedder was not initialized.")
        dev = self._device

        outs: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(texts), max(1, int(batch_size))):
                chunk = texts[i : i + max(1, int(batch_size))]
                batch = self._tokenizer(chunk, padding=True, truncation=True, return_tensors="pt").to(dev)
                out = self._model(**batch)
                # Common HF outputs: pooler_output [B,D] or last_hidden_state [B,S,D]
                vec = getattr(out, "pooler_output", None)
                if vec is None:
                    lhs = getattr(out, "last_hidden_state", None)
                    if lhs is None:
                        raise ValueError("Embedding model output missing pooler_output/last_hidden_state")
                    vec = lhs.mean(dim=1)
                vec = vec.detach().to("cpu")
                if self.normalize:
                    vec = _l2_normalize(vec)
                outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)

    def _embed_remote(self, texts: List[str], *, batch_size: int) -> torch.Tensor:
        if not self._endpoint:
            raise RuntimeError("Remote embedder has no endpoint configured.")

        vecs: List[torch.Tensor] = []
        bs = max(1, int(batch_size))
        with httpx.Client(timeout=self._timeout_seconds) as client:
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                payload = {"input": chunk}
                # Many NIM endpoints are OpenAI-compatible and require/accept "model".
                if self.model_name:
                    payload["model"] = self.model_name

                resp = client.post(self._endpoint, headers=self._headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                emb = _coerce_embeddings_from_response(data)
                if len(emb) != len(chunk):
                    raise RuntimeError(
                        f"Embedding response size mismatch: got {len(emb)} embeddings for {len(chunk)} inputs."
                    )
                t = torch.tensor(emb, dtype=torch.float32)
                if self.normalize:
                    t = _l2_normalize(t)
                vecs.append(t)

        return torch.cat(vecs, dim=0) if vecs else torch.empty((0, 0), dtype=torch.float32)
