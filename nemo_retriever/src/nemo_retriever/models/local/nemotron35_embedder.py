# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native HF/vLLM adapter for dense Mistral3 retrieval checkpoints."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch

from nemo_retriever.models.inference.input_audit import current_rows

if TYPE_CHECKING:
    from nemo_retriever.models.inference.embedding_input import EmbeddingSplitPlan

from nemo_retriever.models.inference.retrieval_template import (
    load_retrieval_template,
    render_retrieval_input,
)


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

    def __post_init__(self) -> None:
        """Validate configuration; raise ValueError for unsupported backend, limits, or dimensions."""
        if self.backend not in {"hf", "vllm"}:
            raise ValueError("Nemotron 3.5 embedding backend must be 'hf' or 'vllm'.")
        if min(self.max_length, self.query_max_length) <= 0:
            raise ValueError("Embedding input limits must be positive.")
        if self.dimensions not in {None, self.output_dimension}:
            raise ValueError("This checkpoint profile supports only its full embedding dimension.")

    @property
    def is_remote(self) -> bool:
        """Return False because this adapter owns local HF or in-process vLLM inference."""
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
            # Ministral3 initializes attention modules as causal even when
            # the checkpoint requests a bidirectional mask. SDPA can omit an
            # all-visible mask, then consult this module flag instead.
            for module in self._model.language_model.modules():
                if hasattr(module, "is_causal"):
                    module.is_causal = False
        else:
            from nemo_retriever.models.inference.nemotron35_vllm import (
                NEMOTRON35_VLLM_ARCHITECTURE,
                register_nemotron35_vllm,
            )
            from nemo_retriever.models.inference.vllm import create_vllm_llm

            register_nemotron35_vllm()
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
                model_impl="vllm",
                hf_overrides={
                    "architectures": [NEMOTRON35_VLLM_ARCHITECTURE],
                    "is_causal": False,
                },
            )

    def _encode(self, texts, *, query=False, images=None, batch_size=64):
        if images is not None and len(images) != len(texts):
            raise ValueError("Text and image counts must match.")
        if not texts:
            return torch.empty((0, self.output_dimension), dtype=torch.float32)
        self._ensure_loaded()
        audit = current_rows()
        if audit is not None:
            audit.sink.model_header(self)
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
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                if int(batch["attention_mask"].sum(1).max()) > limit:
                    raise ValueError(
                        f"Text+image embedding exceeds max_length={limit} after image expansion; "
                        "shorten the text or increase the configured limit within the checkpoint maximum."
                    )
            if audit is not None:
                audit.processor(batch, texts[i : i + step], current, i)
            if self.backend == "hf":
                batch = batch.to(self._model.device)
                # Position real tokens independently of padding side/batch peers.
                positions = batch["attention_mask"].long().cumsum(-1) - 1
                batch["position_ids"] = positions.masked_fill(batch["attention_mask"] == 0, 0)
                if audit is not None:
                    audit.model_call(batch, i, len(current))
                with torch.inference_mode():
                    hidden = self._model(**batch).last_hidden_state.float()
                    mask = batch["attention_mask"].unsqueeze(-1).float()
                    vectors = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
            else:
                from nemo_retriever.models.inference.vllm import (
                    _pooling_params_for_normalize,
                )

                # Raw tokens prevent a second template pass or duplicate BOS/EOS.
                requests = []
                for j in range(len(current)):
                    if images is None:
                        ids = batch["input_ids"][j][batch["attention_mask"][j].bool()].tolist()
                        request = {"prompt_token_ids": ids}
                    else:
                        # vLLM owns image expansion; do not send HF-expanded IDs.
                        request = {
                            "prompt": current[j],
                            "multi_modal_data": {"image": images[i + j]},
                            "mm_processor_kwargs": {
                                "text_kwargs": {"truncation": False, "add_special_tokens": False},
                            },
                        }
                    requests.append(request)
                inference_kwargs = {}
                if images is not None:
                    # Match admission, independent of checkpoint defaults.
                    inference_kwargs["tokenization_kwargs"] = {"add_special_tokens": False, "truncation": False}
                responses = self._model.embed(
                    requests, pooling_params=_pooling_params_for_normalize(False), **inference_kwargs
                )
                if images is not None:
                    expected_lengths = batch["attention_mask"].sum(1).tolist()
                    actual_lengths = [len(response.prompt_token_ids) for response in responses]
                    if actual_lengths != expected_lengths:
                        raise ValueError(
                            "vLLM image token expansion does not match checkpoint processor admission: "
                            f"expected={expected_lengths}, actual={actual_lengths}."
                        )
                vectors = torch.tensor([r.outputs.embedding for r in responses], dtype=torch.float32)
            if vectors.shape != (len(current), self.output_dimension) or not torch.isfinite(vectors).all():
                raise ValueError("Checkpoint returned invalid dense embedding vectors.")
            final = torch.nn.functional.normalize(vectors, dim=-1) if self.normalize else vectors
            if audit is not None:
                audit.outputs(vectors, final, i)
            outputs.append(final)
        return torch.cat(outputs)

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed document text with the checkpoint's document template.

        Parameters
        ----------
        texts : Sequence[str]
            Documents in output order. Formatted text is truncated to max_length.
        batch_size : int
            Requested microbatch size, clamped to at least one; HF caps it at eight.

        Returns
        -------
        torch.Tensor
            CPU float32 vectors of shape (len(texts), output_dimension), including
            an empty tensor for no inputs. Vectors are L2-normalized if normalize is True.

        Raises
        ------
        ValueError
            If the checkpoint returns incorrectly shaped or nonfinite vectors.
        OSError, RuntimeError
            If checkpoint loading or backend inference fails.
        """
        return self._encode([str(t) for t in texts], batch_size=batch_size)

    def embed_queries(self, texts: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed queries with the checkpoint's query template.

        Parameters
        ----------
        texts : Sequence[str]
            Queries in output order. Formatted queries are truncated to query_max_length.
        batch_size : int
            Requested microbatch size, clamped to at least one; HF caps it at eight.

        Returns
        -------
        torch.Tensor
            CPU float32 vectors of shape (len(texts), output_dimension), or an
            empty tensor for no queries. Vectors are normalized when normalize is True.

        Raises
        ------
        ValueError
            If inference returns incorrectly shaped or nonfinite vectors.
        OSError, RuntimeError
            If checkpoint loading or backend inference fails.
        """
        return self._encode([str(t) for t in texts], query=True, batch_size=batch_size)

    def plan_text_image(self, text: str, image_b64: str, *, query: bool = False) -> "EmbeddingSplitPlan":
        """Split paired text losslessly against the exact expanded image budget.

        The caller repeats the unchanged source image and records split
        provenance for every child. Direct embedding methods remain strict:
        they cannot change row cardinality and therefore reject overflow.

        Parameters
        ----------
        text : str
            Source text to preserve across split children.
        image_b64 : str
            Base64-encoded image repeated unchanged with each child.
        query : bool
            Use the query template and query_max_length instead of document settings.

        Returns
        -------
        EmbeddingSplitPlan
            Expanded token count and lossless text children, if splitting is needed.

        Raises
        ------
        ValueError
            If the image and template leave no text budget within the configured limit.
        OSError, RuntimeError
            If image decoding, checkpoint loading, or processor execution fails.
        """
        from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy
        from nemo_retriever.models.inference.image import decode_base64_image

        self._ensure_loaded()
        limit = self.query_max_length if query else self.max_length
        prefix = self.query_prefix if query else self.document_prefix
        image = decode_base64_image(image_b64)
        try:

            def expanded_count(prompt):
                batch = self._processor(
                    text=[prompt],
                    images=[image],
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                return int(batch["attention_mask"].sum())

            policy = EmbeddingInputPolicy(
                tokenizer=self._processor.tokenizer,
                max_tokens=limit,
                prefix=prefix,
                retrieval_template=self._template,
                add_special_tokens=False,
                image=True,
                formatted_token_counter=expanded_count,
            )
            if policy._formatted_token_count("") >= limit:
                raise ValueError(
                    f"Image and retrieval template leave no text budget within max_length={limit}; "
                    "increase the configured limit within the checkpoint maximum. The image is not truncated."
                )
            return policy.plan([text])[0]
        finally:
            image.close()

    def embed_text_image(
        self, texts: Sequence[str], images_b64: Sequence[str], *, batch_size: int = 64
    ) -> torch.Tensor:
        """Embed paired document text and images without silently truncating either.

        Parameters
        ----------
        texts : Sequence[str]
            Document text aligned one-to-one with images_b64.
        images_b64 : Sequence[str]
            Base64-encoded images. Full page images are accepted; the checkpoint
            processor applies its native resizing and expands image tokens.
        batch_size : int
            Requested microbatch size, clamped to at least one; HF caps it at two.

        Returns
        -------
        torch.Tensor
            CPU float32 vectors of shape (len(texts), output_dimension), or an
            empty tensor for no pairs. Vectors are normalized when normalize is True.

        Raises
        ------
        ValueError
            If input counts differ, expanded text/template/image tokens exceed
            max_length, vLLM token expansion differs, or returned vectors are invalid.
            Use plan_text_image for lossless splitting before embedding overlong pairs.
        OSError, RuntimeError
            If image decoding, checkpoint loading, or backend inference fails.
        """
        from nemo_retriever.models.inference.image import decode_base64_image

        images = []
        try:
            for value in images_b64:
                images.append(decode_base64_image(value))
            return self._encode(list(texts), images=images, batch_size=batch_size)
        finally:
            for image in images:
                image.close()

    def embed_images(self, images_b64: Sequence[str], *, batch_size: int = 64) -> torch.Tensor:
        """Embed images using empty document text and the document template.

        Parameters
        ----------
        images_b64 : Sequence[str]
            Base64-encoded images in output order, including full rendered pages.
        batch_size : int
            Requested microbatch size, clamped to at least one; HF caps it at two.

        Returns
        -------
        torch.Tensor
            CPU float32 vectors of shape (len(images_b64), output_dimension), or
            an empty tensor. Vectors are normalized when normalize is True.

        Raises
        ------
        ValueError
            If expanded image/template tokens exceed max_length, vLLM expansion
            differs, or returned vectors are invalid. Images are not silently clipped.
        OSError, RuntimeError
            If image decoding, checkpoint loading, or backend inference fails.
        """
        return self.embed_text_image([""] * len(images_b64), images_b64, batch_size=batch_size)

    def unload(self) -> None:
        """Shut down owned vLLM workers, then release local inference resources.

        vLLM engine shutdown uses a 30-second timeout. Backend shutdown errors
        propagate unchanged rather than allowing a second engine to be loaded.

        Parameters
        ----------
        None
            No arguments are required. Calling this method repeatedly is supported.

        Returns
        -------
        None
            Model, processor, and template references are cleared after successful
            shutdown. Later embedding calls lazily reload the checkpoint.

        Raises
        ------
        RuntimeError
            If engine shutdown or clearing the CUDA allocator cache fails.
            References are retained if shutdown fails so cleanup can be retried.
        """
        if self.backend == "vllm" and self._model is not None:
            self._model.llm_engine.engine_core.shutdown(timeout=30.0)
        self._model = self._processor = self._template = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
