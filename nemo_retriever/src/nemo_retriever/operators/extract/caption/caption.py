# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from nemo_retriever.common.modality.caption.model_profiles import (
    DEFAULT_LOCAL_CAPTION_MODEL_ID,
    get_caption_model_profile,
    merge_request_extras,
    resolve_caption_model_name,
)
from nemo_retriever.common.modality.image.load import SUPPORTED_IMAGE_EXTENSIONS
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.operators.extract.ocr.ocr import _crop_b64_image_by_norm_bbox
from nemo_retriever.common.params import CaptionParams
from nemo_retriever.common.remote_auth import resolve_remote_api_key

_DEFAULT_MODEL_NAME = DEFAULT_LOCAL_CAPTION_MODEL_ID
_DEFAULT_REMOTE_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
_MAX_CONTEXT_TEXT_CHARS = 4096
_MIN_IMAGE_DIMENSION = 32
_LOCAL_MODEL_CACHE_KEYS = (
    "model_name",
    "device",
    "hf_cache_dir",
    "tensor_parallel_size",
    "gpu_memory_utilization",
)
_cached_local_model: dict[tuple[tuple[str, Any], ...], Any] | None = {}


def _image_meets_min_size(b64: str) -> bool:
    """Return True if the base64 image is at least _MIN_IMAGE_DIMENSION on both sides."""
    try:
        img = Image.open(BytesIO(base64.b64decode(b64)))
        w, h = img.size
        return w >= _MIN_IMAGE_DIMENSION and h >= _MIN_IMAGE_DIMENSION
    except Exception:
        return False


def _is_direct_image_row(row: pd.Series) -> bool:
    """Return whether a pipeline row originated from a standalone image file."""
    metadata = row.get("metadata")
    source_path = metadata.get("source_path") if isinstance(metadata, dict) else None
    for candidate in (row.get("path"), source_path):
        if isinstance(candidate, str) and Path(candidate).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            return True
    return False


def _as_image_list(value: Any) -> List[Any] | None:
    """Normalize image collections materialized by pandas or Ray."""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == object:
        return value.tolist()
    return None


def _ensure_object_column(batch_df: pd.DataFrame, column: str) -> None:
    """Make a nested Arrow column accept complete Python collection replacements."""
    if column in batch_df.columns and isinstance(batch_df[column].dtype, pd.ArrowDtype):
        batch_df[column] = batch_df[column].astype(object)


def _write_caption(
    batch_df: pd.DataFrame,
    *,
    row_idx: Any,
    column: str,
    item_idx: int,
    field: str,
    caption: str,
) -> None:
    """Persist a caption by replacing the complete nested collection cell."""
    items = _as_image_list(batch_df.at[row_idx, column])
    if items is None or item_idx >= len(items) or not isinstance(items[item_idx], dict):
        return
    updated_items = [dict(item) if isinstance(item, dict) else item for item in items]
    updated_items[item_idx][field] = caption
    batch_df.at[row_idx, column] = updated_items


def _create_local_model(kwargs: dict) -> "Any":
    from nemo_retriever.models.local import NemotronVLMCaptioner

    return NemotronVLMCaptioner(
        model_path=kwargs.get("model_name", _DEFAULT_MODEL_NAME),
        device=kwargs.get("device"),
        hf_cache_dir=kwargs.get("hf_cache_dir"),
        max_new_tokens=kwargs.get("max_tokens", 1024),
        tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.5),
    )


def _local_model_cache_key(kwargs: dict) -> tuple[tuple[str, Any], ...]:
    resolved_kwargs = {
        "model_name": kwargs.get("model_name", _DEFAULT_MODEL_NAME),
        "device": kwargs.get("device"),
        "hf_cache_dir": kwargs.get("hf_cache_dir"),
        "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.5),
    }
    return tuple((key, resolved_kwargs[key]) for key in _LOCAL_MODEL_CACHE_KEYS)


def _get_cached_local_model(kwargs: dict) -> "Any":
    global _cached_local_model
    if _cached_local_model is None:
        _cached_local_model = {}

    cache_key = _local_model_cache_key(kwargs)
    if cache_key not in _cached_local_model:
        _cached_local_model[cache_key] = _create_local_model(kwargs)
    return _cached_local_model[cache_key]


@designer_component(
    name="Image Captioner",
    category="Embeddings & Ranking",
    compute="gpu",
    description="Generates captions for images using a vision-language model",
    category_color="#e06cff",
)
class CaptionGPUActor(AbstractOperator, GPUOperator):
    """Ray Data actor that holds a local VLM captioner on a single GPU."""

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
        self._kwargs = params.model_dump(mode="python")
        endpoint = (self._kwargs.get("endpoint_url") or "").strip()
        if endpoint:
            raise ValueError("CaptionGPUActor does not support remote endpoint execution. Use CaptionCPUActor instead.")
        self._model = _create_local_model(self._kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        return caption_images(batch_df, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class CaptionCPUActor(AbstractOperator, CPUOperator):
    """CPU-only caption actor that delegates to a remote VLM endpoint."""

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params
        self._kwargs = params.model_dump(mode="python")
        configured_endpoint = (self._kwargs.get("endpoint_url") or "").strip()
        endpoint = configured_endpoint or _DEFAULT_REMOTE_ENDPOINT_URL
        api_key = resolve_remote_api_key(str(self._kwargs.get("api_key") or ""))
        if api_key:
            self._kwargs["api_key"] = api_key
        if not configured_endpoint and not api_key:
            raise ValueError(
                "CaptionCPUActor defaulted to the hosted VLM endpoint but no API key is configured. "
                "Set NVIDIA_API_KEY/NGC_API_KEY, pass --caption-invoke-url for a local endpoint, "
                "or run on a GPU host for local captioning."
            )
        self._kwargs["endpoint_url"] = endpoint
        self._model = None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **kwargs: Any) -> Any:
        return caption_images(batch_df, model=self._model, **self._kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class CaptionActor(ArchetypeOperator):
    """Graph-facing captioning archetype resolved to the local hardware variant."""

    _cpu_variant_class = CaptionCPUActor
    _gpu_variant_class = CaptionGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        params = (operator_kwargs or {}).get("params")
        endpoint = getattr(params, "endpoint_url", None)
        return bool(str(endpoint or "").strip())

    def __init__(self, params: CaptionParams) -> None:
        super().__init__(params=params)
        self._params = params


def _build_prompt_with_context(base_prompt: str, context_text: str) -> str:
    """Prepend surrounding page text to the base VLM prompt.

    If *context_text* is empty the *base_prompt* is returned unchanged.
    """
    if not context_text:
        return base_prompt
    return f"Text near this image:\n---\n{context_text}\n---\n\n{base_prompt}"


def _create_remote_client(endpoint_url: str, api_key: str | None) -> Any:
    """Create a reusable NIM inference client for a remote VLM endpoint."""
    from nemo_retriever.models.nim.primitives.model_interface.vlm import VLMModelInterface
    from nemo_retriever.models.nim.util import create_inference_client

    return create_inference_client(
        model_interface=VLMModelInterface(),
        endpoints=(None, endpoint_url),
        auth_token=api_key,
        infer_protocol="http",
    )


def _caption_batch_remote(
    base64_images: List[str],
    *,
    nim_client: Any,
    model_name: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    top_p: float | None = None,
    max_tokens: int = 1024,
    request_extras: Dict[str, Any] | None = None,
) -> List[str]:
    """Send a batch of images to a remote VLM endpoint and return captions."""
    from nemo_retriever.common.api.util.image_processing.transforms import scale_image_to_encoding_size

    scaled = [scale_image_to_encoding_size(b64)[0] for b64 in base64_images]

    data: Dict[str, Any] = {
        "base64_images": scaled,
        "prompt": prompt,
    }
    if system_prompt:
        data["system_prompt"] = system_prompt

    from nemo_retriever.common.params.models import LLMInferenceParams

    sampling_kwargs = LLMInferenceParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ).to_sampling_kwargs()
    infer_kwargs: Dict[str, Any] = merge_request_extras(
        sampling_kwargs,
        request_extras or {},
    )
    infer_kwargs["model_name"] = model_name

    return nim_client.infer(data, **infer_kwargs)


def _caption_batch_local(
    base64_images: List[str],
    *,
    model: Any,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    top_p: float | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> List[str]:
    """Generate captions using a local ``NemotronVLMCaptioner`` model."""
    caption_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if extra_body:
        caption_kwargs["extra_body"] = extra_body
    return model.caption_batch(base64_images, **caption_kwargs)


def _caption_one(
    b64: str,
    *,
    model: Any,
    nim_client: Any | None,
    model_name: str,
    prompt: str,
    system_prompt: str | None,
    temperature: float,
    top_p: float | None = None,
    max_tokens: int | None = None,
    request_extras: Dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Caption a single image (used when each image gets a unique prompt)."""
    if model is not None:
        captions = _caption_batch_local(
            [b64],
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    else:
        captions = _caption_batch_remote(
            [b64],
            nim_client=nim_client,
            model_name=model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            request_extras=request_extras,
        )
    return captions[0] if captions else ""


def caption_images(
    batch_df: pd.DataFrame,
    *,
    model: Any = None,
    endpoint_url: str | None = None,
    model_name: str = _DEFAULT_MODEL_NAME,
    api_key: str | None = None,
    prompt: str = "Caption the content of this image:",
    system_prompt: str | None = "/no_think",
    temperature: float = 1.0,
    top_p: float | None = None,
    max_tokens: int = 1024,
    batch_size: int = 8,
    context_text_max_chars: int = 0,
    caption_infographics: bool = False,
    extra_body: dict[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Caption images in the ``images`` column using a VLM.

    Supports two modes:

    * **Remote** (``endpoint_url`` is set): sends images to an HTTP VLM
      endpoint via ``create_inference_client`` / ``VLMModelInterface``.
    * **Local** (``model`` is set): runs inference through a local
      ``NemotronVLMCaptioner`` instance loaded from Hugging Face.

    When ``context_text_max_chars`` is greater than zero, the page's ``text``
    column is prepended to the prompt for each image so the VLM can use
    surrounding OCR text as context.  In this mode images are captioned
    one at a time (each gets its own enriched prompt).

    For each row, any item in the ``images`` list whose ``text`` field is
    empty will be captioned.  The returned caption is written back into
    ``images[i]["text"]``.

    For a standalone image row with an empty ``images`` list, the full
    ``page_image`` is materialized as an image entry before captioning.

    When ``caption_infographics`` is True, infographic entries are cropped
    from the page image and captioned.  The VLM caption is written to the
    ``caption`` field, preserving the existing OCR ``text``.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    has_images = "images" in batch_df.columns
    has_page_image = "page_image" in batch_df.columns
    has_direct_page_image = has_page_image and any(_is_direct_image_row(row) for _, row in batch_df.iterrows())
    has_infographics = caption_infographics and "infographic" in batch_df.columns
    if not has_images and not has_direct_page_image and not has_infographics:
        return batch_df
    if not has_images and has_direct_page_image:
        batch_df["images"] = [[] for _ in range(len(batch_df))]
        has_images = True
    if has_images:
        _ensure_object_column(batch_df, "images")
    if has_infographics:
        _ensure_object_column(batch_df, "infographic")

    request_extras: Dict[str, Any] = {}
    if endpoint_url:
        profile = get_caption_model_profile(model_name, target="remote", strict=False)
        if profile is not None:
            model_name = profile.remote_model_id or model_name
            request_extras = profile.request_extras_for("remote")
        request_extras = merge_request_extras(request_extras, extra_body or {})
        if extra_body:
            # VLMModelInterface.format_input handles only chat_template_kwargs, mm_processor_kwargs, and
            # media_options as first-class kwargs. Pack merged extras here so arbitrary caller keys
            # such as custom_request_id reach the final payload via payload.update(extra_body).
            request_extras["extra_body"] = merge_request_extras(
                {},
                {key: value for key, value in request_extras.items() if key != "extra_body"},
            )
    else:
        model_name = resolve_caption_model_name(model_name, target="local")

    if model is None and not endpoint_url:
        local_kwargs = dict(kwargs)
        local_kwargs["model_name"] = model_name
        local_kwargs["max_tokens"] = max_tokens
        model = _get_cached_local_model(local_kwargs)

    nim_client = _create_remote_client(endpoint_url, api_key) if endpoint_url and model is None else None

    use_context = context_text_max_chars > 0
    effective_max = min(context_text_max_chars, _MAX_CONTEXT_TEXT_CHARS) if use_context else 0

    # pending entries: (row_idx, column_name, item_idx, b64)
    pending: List[Tuple[int, str, int, str]] = []
    for row_idx, row in batch_df.iterrows():
        # Unstructured images.
        if has_images:
            images = _as_image_list(row.get("images"))
            if images is not None:
                if not images and _is_direct_image_row(row):
                    page_image = row.get("page_image")
                    page_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
                    if page_b64 and _image_meets_min_size(page_b64):
                        image_entry = {
                            "image_b64": page_b64,
                            "text": "",
                            "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0],
                        }
                        batch_df.at[row_idx, "images"] = [image_entry]
                        pending.append((row_idx, "images", 0, page_b64))
                    continue
                for item_idx, item in enumerate(images):
                    if not isinstance(item, dict):
                        continue
                    if item.get("text"):
                        continue
                    b64 = item.get("image_b64")
                    if b64 and _image_meets_min_size(b64):
                        pending.append((row_idx, "images", item_idx, b64))

        # Infographics — crop from page image.
        if has_infographics:
            infographics = _as_image_list(row.get("infographic"))
            if infographics is not None:
                page_image = row.get("page_image")
                page_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
                if page_b64:
                    for item_idx, item in enumerate(infographics):
                        if not isinstance(item, dict):
                            continue
                        if item.get("caption"):
                            continue  # already captioned
                        bbox = item.get("bbox_xyxy_norm")
                        if bbox is None or len(bbox) < 4:
                            continue
                        cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_b64, bbox_xyxy_norm=bbox)
                        if cropped_b64 and _image_meets_min_size(cropped_b64):
                            pending.append((row_idx, "infographic", item_idx, cropped_b64))

    if not pending:
        return batch_df

    if use_context:
        for row_idx, col, item_idx, b64 in pending:
            page_text = batch_df.at[row_idx, "text"] if "text" in batch_df.columns else ""
            context = (page_text or "")[:effective_max]
            enriched_prompt = _build_prompt_with_context(prompt, context)
            caption = _caption_one(
                b64,
                model=model,
                nim_client=nim_client,
                model_name=model_name,
                prompt=enriched_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                request_extras=request_extras,
                extra_body=extra_body,
            )
            # Infographics keep OCR text; VLM caption goes to a separate field.
            field = "caption" if col == "infographic" else "text"
            _write_caption(
                batch_df,
                row_idx=row_idx,
                column=col,
                item_idx=item_idx,
                field=field,
                caption=caption,
            )
    else:
        all_b64 = [b64 for _, _, _, b64 in pending]

        if model is not None:
            all_captions = _caption_batch_local(
                all_b64,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        else:
            all_captions: List[str] = []
            for start in range(0, len(all_b64), batch_size):
                captions = _caption_batch_remote(
                    all_b64[start : start + batch_size],
                    nim_client=nim_client,
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    request_extras=request_extras,
                )
                all_captions.extend(captions)

        for (row_idx, col, item_idx, _), caption in zip(pending, all_captions):
            field = "caption" if col == "infographic" else "text"
            _write_caption(
                batch_df,
                row_idx=row_idx,
                column=col,
                item_idx=item_idx,
                field=field,
                caption=caption,
            )

    return batch_df
