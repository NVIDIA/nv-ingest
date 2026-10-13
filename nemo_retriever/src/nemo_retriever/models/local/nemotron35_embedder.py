# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native HF/vLLM adapter for dense Mistral3 retrieval checkpoints."""
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from nemo_retriever.models.inference.retrieval_template import load_retrieval_template, render_retrieval_input


@dataclass
class Nemotron35Embedder:
    model_id: str
    revision: str | None
    query_prefix: str
    document_prefix: str
    backend: str = "vllm"
    device: str | None = None
    hf_cache_dir: str | None = None
    normalize: bool = True
    output_dimension: int = 2048
    dimensions: int | None = None
    max_length: int = 8192
    query_max_length: int = 128
    gpu_memory_utilization: float = 0.45
    enforce_eager: bool = False
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _template: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.backend not in {"hf", "vllm"}:
            raise ValueError("Nemotron 3.5 embedding backend must be 'hf' or 'vllm'.")
        if min(self.max_length, self.query_max_length) <= 0:
            raise ValueError("Embedding input limits must be positive.")
        if self.dimensions not in {None, self.output_dimension}:
            raise ValueError("This checkpoint profile supports only its full embedding dimension.")

    @property
    def is_remote(self):
        return False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        options = dict(revision=self.revision, cache_dir=self.hf_cache_dir)
        self._template = load_retrieval_template(self.model_id, self.revision, self.hf_cache_dir)
        self._processor = AutoProcessor.from_pretrained(self.model_id, **options)
        if self.backend == "hf":
            self._model = (
                AutoModel.from_pretrained(
                    self.model_id,
                    **options,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                .to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                .eval()
            )
        else:
            from nemo_retriever.models.inference.vllm import create_vllm_llm

            # Include room for processor-expanded images; per-request limits are
            # validated separately. Do not reinterpret this as a causal LLM.
            self._model = create_vllm_llm(
                self.model_id,
                revision=self.revision,
                max_model_len=max(self.max_length, self.query_max_length),
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                limit_mm_per_prompt={"image": 1},
                download_dir=self.hf_cache_dir,
            )

    def _encode(self, texts, *, query=False, images=None, batch_size=64):
        if images is not None and len(images) != len(texts):
            raise ValueError("Text and image counts must match.")
        if not texts:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        self._ensure_loaded()
        prefix = self.query_prefix if query else self.document_prefix
        limit = self.query_max_length if query else self.max_length
        prompts = [render_retrieval_input(self._template, prefix, text, image=images is not None) for text in texts]
        step = max(1, int(batch_size))
        if self.backend == "hf":
            step = min(step, 2 if images is not None else 8)
        outputs = []
        for i in range(0, len(prompts), step):
            current = prompts[i : i + step]
            if images is None:
                batch = self._processor.tokenizer(
                    current,
                    padding=True,
                    truncation=True,
                    max_length=limit,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            else:
                batch = self._processor(
                    text=current,
                    images=images[i : i + step],
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                if int(batch["attention_mask"].sum(1).max()) > limit:
                    raise ValueError(
                        f"Text+image embedding exceeds max_length={limit} after image expansion; "
                        "shorten the text or increase the configured limit within the checkpoint maximum."
                    )
            if self.backend == "hf":
                batch = batch.to(self._model.device)
                # Position real tokens independently of padding side/batch peers.
                positions = batch["attention_mask"].long().cumsum(-1) - 1
                batch["position_ids"] = positions.masked_fill(batch["attention_mask"] == 0, 0)
                with torch.inference_mode():
                    hidden = self._model(**batch).last_hidden_state.float()
                    mask = batch["attention_mask"].unsqueeze(-1).float()
                    vectors = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
            else:
                from nemo_retriever.models.inference.vllm import _pooling_params_for_normalize

                # Raw tokens prevent a second template pass or duplicate BOS/EOS.
                requests = []
                for j in range(len(current)):
                    if images is None:
                        ids = batch["input_ids"][j][batch["attention_mask"][j].bool()].tolist()
                        request = {"prompt_token_ids": ids}
                    else:
                        # vLLM owns image expansion; do not send HF-expanded IDs.
                        request = {"prompt": current[j], "multi_modal_data": {"image": images[i + j]}}
                    requests.append(request)
                responses = self._model.embed(requests, pooling_params=_pooling_params_for_normalize(False))
                vectors = torch.tensor([r.outputs.embedding for r in responses], dtype=torch.float32)
            if vectors.shape != (len(current), self.output_dimension) or not torch.isfinite(vectors).all():
                raise ValueError("Checkpoint returned invalid dense embedding vectors.")
            outputs.append(torch.nn.functional.normalize(vectors, dim=-1) if self.normalize else vectors)
        return torch.cat(outputs)

    def embed(self, texts: Sequence[str], *, batch_size=64):
        return self._encode([str(t) for t in texts], batch_size=batch_size)

    def embed_queries(self, texts: Sequence[str], *, batch_size=64):
        return self._encode([str(t) for t in texts], query=True, batch_size=batch_size)

    def embed_text_image(self, texts, images_b64, *, batch_size=64):
        from nemo_retriever.models.local.llama_nemotron_embed_vl_1b_v2_embedder import _b64_to_pil

        images = []
        try:
            for value in images_b64:
                images.append(_b64_to_pil(value))
            return self._encode(list(texts), images=images, batch_size=batch_size)
        finally:
            for image in images:
                image.close()

    def embed_images(self, images_b64, *, batch_size=64):
        return self.embed_text_image([""] * len(images_b64), images_b64, batch_size=batch_size)

    def unload(self):
        self._model = self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
