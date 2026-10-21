# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Parse pipeline stage for native multipart and legacy contracts.

Runs the Nemotron Parse model on full page images to extract structured
document content (text, tables, charts, infographics) in a single pass,
replacing the page-elements → OCR multi-stage pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import base64
import io
import json
import re
import time
import traceback

import numpy as np
import pandas as pd

from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import (
    extract_classes_bboxes,
    postprocess_text as _postprocess_element_text,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.models.nim.chat_completions import invoke_chat_completions_images
from nemo_retriever.models.nim.nim import NIMClient, invoke_image_inference_batches
from nemo_retriever.common.params import RemoteRetryParams
from nemo_retriever.common.params.utils import (
    NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL,
    NEMOTRON_PARSE_LOCAL_MODELS,
    NEMOTRON_PARSE35_MODEL,
    validate_nemotron_parse_endpoint_list,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEMOTRON_PARSE_REMOTE_DEFAULT_MODEL = "nvidia/nemotron-parse-v1.2"
NEMOTRON_PARSE_HOSTED_MODEL = "nvidia/nemotron-parse"
NEMOTRON_PARSE_DEFAULT_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

# Map Nemotron Parse class labels to the pipeline content channels.
_PARSE_CLASS_TO_CHANNEL: Dict[str, str] = {
    "Table": "table",
    "Chart": "chart",
    "Picture": "infographic",
    "Infographic": "infographic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "timing": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


def _extract_parse_text(response_item: Any) -> str:
    """Extract text from a Nemotron Parse NIM response item."""
    if response_item is None:
        return ""
    if isinstance(response_item, str):
        return response_item.strip()
    if isinstance(response_item, dict):
        for key in ("generated_text", "text", "output_text", "prediction", "output", "data"):
            value = response_item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
                if isinstance(first, dict):
                    inner = _extract_parse_text(first)
                    if inner:
                        return inner
    if isinstance(response_item, list):
        for item in response_item:
            text = _extract_parse_text(item)
            if text:
                return text
    try:
        return str(response_item).strip()
    except Exception:
        return ""


def _route_parsed_elements(
    raw_text: str,
    *,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Split raw Nemotron Parse output by element class into pipeline channels.

    Returns ``(table_items, chart_items, infographic_items, page_text)``
    where each ``*_items`` list contains ``{"bbox_xyxy_norm": ..., "text": ...}``
    dicts and ``page_text`` is the concatenated text of non-structured elements
    (or ``None`` if there are none).
    """
    classes, bboxes, texts = extract_classes_bboxes(raw_text)
    table_items: List[Dict[str, Any]] = []
    chart_items: List[Dict[str, Any]] = []
    infographic_items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for cls, bbox, text in zip(classes, bboxes, texts):
        bbox_list = list(bbox)
        processed = _postprocess_element_text(text, cls=cls, table_format="markdown")
        if not processed:
            continue
        channel = _PARSE_CLASS_TO_CHANNEL.get(cls)
        entry = {"bbox_xyxy_norm": bbox_list, "text": processed}
        if channel == "table" and extract_tables:
            table_items.append(entry)
        elif channel == "chart" and extract_charts:
            chart_items.append(entry)
        elif channel == "infographic" and extract_infographics:
            infographic_items.append(entry)
        else:
            # Text, Title, Header_footer, Formula, etc. → page text
            text_parts.append(processed)

    page_text = "\n\n".join(text_parts) if text_parts else None
    return table_items, chart_items, infographic_items, page_text


class _NemotronParseContractProfile(str, Enum):
    HOSTED_TOOL_CALL = "hosted_tool_call"
    LEGACY_TOOL_CALL = "legacy_tool_call"
    V1_2_TAGGED = "v1_2_tagged"


@dataclass(frozen=True)
class _ResolvedNemotronParseContract:
    model: str
    profile: _NemotronParseContractProfile
    has_build_endpoint: bool

    @property
    def uses_tool_call_routing(self) -> bool:
        return self.profile in {
            _NemotronParseContractProfile.HOSTED_TOOL_CALL,
            _NemotronParseContractProfile.LEGACY_TOOL_CALL,
        }


def _is_legacy_nemotron_parse_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return bool(re.search(r"v1[._][01](?!\d)", normalized))


def _is_nvidia_build_endpoint(invoke_url: str) -> bool:
    return (urlsplit(invoke_url).hostname or "").lower() == "integrate.api.nvidia.com"


def _resolve_nemotron_parse_contract(
    invoke_url: str,
    model_name: Optional[str],
) -> _ResolvedNemotronParseContract:
    """Resolve the internal request/response contract for a chat endpoint."""

    invoke_urls = validate_nemotron_parse_endpoint_list(invoke_url)
    if not invoke_urls:
        raise ValueError("Nemotron Parse invoke_url is required.")

    build_endpoints = [_is_nvidia_build_endpoint(url) for url in invoke_urls]
    explicit_model = str(model_name or "").strip()
    resolved_model = explicit_model or (
        NEMOTRON_PARSE_HOSTED_MODEL if all(build_endpoints) else NEMOTRON_PARSE_REMOTE_DEFAULT_MODEL
    )
    normalized_model = resolved_model.lower()
    if normalized_model == NEMOTRON_PARSE_HOSTED_MODEL:
        profile = _NemotronParseContractProfile.HOSTED_TOOL_CALL
    elif _is_legacy_nemotron_parse_model(normalized_model):
        profile = _NemotronParseContractProfile.LEGACY_TOOL_CALL
    else:
        profile = _NemotronParseContractProfile.V1_2_TAGGED

    return _ResolvedNemotronParseContract(
        model=resolved_model,
        profile=profile,
        has_build_endpoint=any(build_endpoints),
    )


def _route_tool_call_elements(
    raw_json_text: str,
    *,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Route hosted or legacy tool-call JSON into pipeline content channels."""

    try:
        parsed = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        return [], [], [], None

    elements: List[Dict[str, Any]] = []

    def _collect(value: Any) -> None:
        if isinstance(value, dict):
            if "type" in value and ("text" in value or "bbox" in value):
                elements.append(value)
                return
            for nested in value.values():
                _collect(nested)
        elif isinstance(value, list):
            for nested in value:
                _collect(nested)

    _collect(parsed)

    table_items: List[Dict[str, Any]] = []
    chart_items: List[Dict[str, Any]] = []
    infographic_items: List[Dict[str, Any]] = []
    text_parts: List[str] = []

    for elem in elements:
        cls = str(elem.get("type", ""))
        raw_text = str(elem.get("text", "")).strip()
        if not raw_text:
            continue
        text = _postprocess_element_text(raw_text, cls=cls, table_format="markdown")
        if not text:
            continue
        bbox = elem.get("bbox", {})
        bbox_list = [
            float(bbox.get("xmin", 0)),
            float(bbox.get("ymin", 0)),
            float(bbox.get("xmax", 0)),
            float(bbox.get("ymax", 0)),
        ]
        channel = _PARSE_CLASS_TO_CHANNEL.get(cls)
        entry = {"bbox_xyxy_norm": bbox_list, "text": text}
        if channel == "table" and extract_tables:
            table_items.append(entry)
        elif channel == "chart" and extract_charts:
            chart_items.append(entry)
        elif channel == "infographic" and extract_infographics:
            infographic_items.append(entry)
        else:
            text_parts.append(text)

    page_text = "\n\n".join(text_parts) if text_parts else None
    return table_items, chart_items, infographic_items, page_text


def _decode_page_image(page_image_b64: str) -> np.ndarray:
    """Decode a base64 page image to an HWC uint8 numpy array."""
    raw = base64.b64decode(page_image_b64)
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Main stage function
# ---------------------------------------------------------------------------


def nemotron_parse_pages(
    batch_df: Any,
    *,
    model: Any = None,
    invoke_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_timeout_s: float = 120.0,
    extract_text: bool = False,
    extract_tables: bool = False,
    extract_charts: bool = False,
    extract_infographics: bool = False,
    nemotron_parse_model: Optional[str] = None,
    task_prompt: str = NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
    remote_retry: RemoteRetryParams | None = None,
    nim_client: NIMClient | None = None,
    **kwargs: Any,
) -> Any:
    """Run Nemotron Parse on full page images.

    Each page is parsed in a single model call.  The structured output is
    split by element class (Text, Table, Chart, Picture, …) and routed to
    the corresponding pipeline content columns (``table``, ``chart``,
    ``infographic``).  Non-structured elements (headings, body text, …) are
    concatenated into the ``text`` column, replacing the upstream pdfium
    extraction when ``extract_text`` is ``True``.

    Native ``/v1/parse`` endpoints return tagged content in ``content``, which
    uses the same element routing. Untagged Markdown is preserved as page text.
    """
    retry = remote_retry or RemoteRetryParams(
        remote_max_pool_workers=int(kwargs.get("remote_max_pool_workers", 16)),
        remote_max_retries=int(kwargs.get("remote_max_retries", 10)),
        remote_max_429_retries=int(kwargs.get("remote_max_429_retries", 5)),
    )
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("nemotron_parse_pages currently only supports pandas.DataFrame input.")

    invoke_url = str(invoke_url or "").strip() or str(kwargs.get("nemotron_parse_invoke_url") or "").strip()
    use_remote = bool(invoke_url)
    if use_remote:
        validate_nemotron_parse_endpoint_list(invoke_url)
    if not use_remote and model is None:
        raise ValueError("A local `model` is required when `invoke_url` is not provided.")

    n_rows = len(batch_df)
    all_table: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_chart: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_infographic: List[List[Dict[str, Any]]] = [[] for _ in range(n_rows)]
    all_text: List[Optional[str]] = [None] * n_rows
    all_meta: List[Dict[str, Any]] = [{"timing": None, "error": None} for _ in range(n_rows)]

    t0_total = time.perf_counter()

    # -- Phase 1: collect page images that need inference ----------------
    batch_indices: List[int] = []  # index into batch_df
    batch_images: List[Any] = []  # numpy arrays (local) or b64 strings (remote)

    for idx, row in enumerate(batch_df.itertuples(index=False)):
        page_image = getattr(row, "page_image", None) or {}
        page_image_b64 = page_image.get("image_b64") if isinstance(page_image, dict) else None
        if not isinstance(page_image_b64, str) or not page_image_b64:
            continue
        try:
            if use_remote:
                batch_images.append(page_image_b64)
            else:
                batch_images.append(_decode_page_image(page_image_b64))
            batch_indices.append(idx)
        except Exception as e:
            all_meta[idx] = {
                "timing": None,
                "error": {
                    "stage": "nemotron_parse_pages_decode",
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                },
            }

    # -- Phase 2: run model inference in a single batch ------------------
    raw_texts: List[str] = [""] * len(batch_indices)
    uses_tool_call_routing = False
    contract: _ResolvedNemotronParseContract | None = None
    native_parse = False
    if batch_images:
        try:
            if use_remote:
                endpoints = validate_nemotron_parse_endpoint_list(invoke_url)
                native_endpoints = [urlsplit(url).path.rstrip("/") == "/v1/parse" for url in endpoints]
                if any(native_endpoints) and not all(native_endpoints):
                    raise ValueError("Do not mix native /v1/parse and legacy Parse endpoint contracts.")
                native_parse = all(native_endpoints)
                if native_parse:
                    client = nim_client or NIMClient(max_pool_workers=int(retry.remote_max_pool_workers))
                    try:
                        responses = client.invoke_parse_pages(
                            invoke_url=invoke_url,
                            image_b64_list=batch_images,
                            api_key=api_key,
                            timeout_s=float(request_timeout_s),
                            max_retries=int(retry.remote_max_retries),
                            max_429_retries=int(retry.remote_max_429_retries),
                        )
                    finally:
                        if nim_client is None:
                            client.shutdown()
                    raw_texts = [response["content"].strip() for response in responses]
                    for idx, response in zip(batch_indices, responses):
                        all_meta[idx]["native_parse"] = {
                            "finish_reason": response.get("finish_reason"),
                            "usage": response.get("usage"),
                        }
                elif "/v1/chat/completions" in invoke_url:
                    contract = _resolve_nemotron_parse_contract(invoke_url, nemotron_parse_model)
                    uses_tool_call_routing = contract.uses_tool_call_routing
                    extra_body: Dict[str, Any] = {"max_tokens": 8192}
                    if contract.profile == _NemotronParseContractProfile.LEGACY_TOOL_CALL:
                        extra_body["tools"] = [{"type": "function", "function": {"name": "markdown_bbox"}}]
                    _chat_kw = dict(
                        invoke_url=invoke_url,
                        image_b64_list=batch_images,
                        model=contract.model,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        task_prompt=None if uses_tool_call_routing else task_prompt,
                        repetition_penalty=(
                            None if contract.profile == _NemotronParseContractProfile.HOSTED_TOOL_CALL else 1.1
                        ),
                        extra_body=extra_body,
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if nim_client is not None:
                        raw_texts = nim_client.invoke_chat_completions_images(**_chat_kw)
                    else:
                        raw_texts = invoke_chat_completions_images(
                            **_chat_kw,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                        )
                else:
                    _img_kw = dict(
                        invoke_url=invoke_url,
                        image_b64_list=batch_images,
                        api_key=api_key,
                        timeout_s=float(request_timeout_s),
                        max_batch_size=int(kwargs.get("inference_batch_size", 8)),
                        max_retries=int(retry.remote_max_retries),
                        max_429_retries=int(retry.remote_max_429_retries),
                    )
                    if nim_client is not None:
                        response_items = nim_client.invoke_image_inference_batches(**_img_kw)
                    else:
                        response_items = invoke_image_inference_batches(
                            **_img_kw,
                            max_pool_workers=int(retry.remote_max_pool_workers),
                        )
                    raw_texts = [_extract_parse_text(item) for item in response_items]
            else:
                # Local models use task_prompt and return tagged text.
                invoke_batch = getattr(model, "invoke_batch", None)
                if invoke_batch is not None:
                    raw_texts = [str(t or "").strip() for t in invoke_batch(batch_images, task_prompt=task_prompt)]
                else:
                    raw_texts = [str(model.invoke(img, task_prompt=task_prompt) or "").strip() for img in batch_images]
        except BaseException as e:
            if (
                contract is not None
                and nemotron_parse_model
                and contract.has_build_endpoint
                and contract.profile == _NemotronParseContractProfile.V1_2_TAGGED
                and "text input" in str(e).lower()
            ):
                hint = ValueError(
                    "Nemotron Parse model/contract mismatch: NVIDIA Build model "
                    "`nvidia/nemotron-parse` uses an image-only tool-call contract, but "
                    f"`{contract.model}` selected the v1.2 text-control-token contract. "
                    "Use `nemotron_parse_model='nvidia/nemotron-parse'` with Build, or send "
                    "the versioned v1.2 model to a compatible self-hosted endpoint."
                )
                hint.__cause__ = e
                e = hint
            print(f"Warning: Nemotron Parse batch failed: {type(e).__name__}: {e}")
            err = {
                "stage": "nemotron_parse_pages",
                "type": e.__class__.__name__,
                "message": str(e),
                "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            }
            for i in batch_indices:
                all_meta[i] = {"timing": None, "error": err}
            raw_texts = []

    # -- Phase 3: route parsed elements into content channels ------------
    route_fn = _route_tool_call_elements if uses_tool_call_routing else _route_parsed_elements
    for pos, raw_text in enumerate(raw_texts):
        idx = batch_indices[pos]
        try:
            if native_parse and not extract_classes_bboxes(raw_text)[0]:
                # Only complete bounding-box/class sequences select the tagged
                # router. Literal or incomplete tags remain page Markdown.
                if not extract_text:
                    raise ValueError("Untagged native Parse content requires extract_text=True.")
                all_text[idx] = raw_text
                continue
            fp_tables, fp_charts, fp_infographics, fp_text = route_fn(
                raw_text,
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                extract_infographics=extract_infographics,
            )
            all_table[idx] = fp_tables
            all_chart[idx] = fp_charts
            all_infographic[idx] = fp_infographics
            if fp_text is not None:
                all_text[idx] = fp_text
        except BaseException as e:
            all_meta[idx] = {
                "timing": None,
                "error": {
                    "stage": "nemotron_parse_pages_route",
                    "type": e.__class__.__name__,
                    "message": str(e),
                    "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                },
            }

    elapsed = time.perf_counter() - t0_total
    for meta in all_meta:
        meta["timing"] = {"seconds": float(elapsed)}

    out = batch_df.copy()
    if extract_text and "text" in out.columns:
        for i, parse_text in enumerate(all_text):
            if parse_text is not None:
                out.iat[i, out.columns.get_loc("text")] = parse_text
    elif extract_text:
        out["text"] = [t if t is not None else "" for t in all_text]
    out["table"] = all_table
    out["chart"] = all_chart
    out["infographic"] = all_infographic
    out["table_parse"] = all_table
    out["chart_parse"] = all_chart
    out["infographic_parse"] = all_infographic
    out["nemotron_parse_v1_2"] = all_meta
    return out


# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


class NemotronParseGPUActor(AbstractOperator, GPUOperator):
    """Ray-friendly callable that initialises the selected local Parse model once per actor."""

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        nemotron_parse_model: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._invoke_url = str(nemotron_parse_invoke_url or "").strip() or str(invoke_url or "").strip()
        self._nemotron_parse_model = nemotron_parse_model
        validate_nemotron_parse_endpoint_list(self._invoke_url)
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._model = None
        if self._invoke_url:
            self._nim_client = NIMClient(max_pool_workers=int(remote_max_pool_workers))
        else:
            self._nim_client = None
        self._extract_text = bool(extract_text)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def _ensure_model(self) -> None:
        """Load the selected local model on the worker rather than the driver."""
        if self._model is None and not self._invoke_url:
            model_id = self._nemotron_parse_model or NEMOTRON_PARSE_LOCAL_DEFAULT_MODEL
            if model_id not in NEMOTRON_PARSE_LOCAL_MODELS:
                raise ValueError(f"Unsupported local Nemotron Parse model: {model_id}")
            if model_id == NEMOTRON_PARSE35_MODEL:
                from nemo_retriever.models.local import NemotronParse35

                self._model = NemotronParse35(task_prompt=self._task_prompt, backend="vllm")
            else:
                from nemo_retriever.models.local import NemotronParseV12

                self._model = NemotronParseV12(task_prompt=self._task_prompt)

    def process(self, data: Any, **kwargs: Any) -> Any:
        self._ensure_model()
        return nemotron_parse_pages(
            data,
            model=self._model,
            invoke_url=self._invoke_url,
            nemotron_parse_model=self._nemotron_parse_model,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            task_prompt=self._task_prompt,
            extract_text=self._extract_text,
            extract_tables=self._extract_tables,
            extract_charts=self._extract_charts,
            extract_infographics=self._extract_infographics,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=e)}]


class NemotronParseCPUActor(AbstractOperator, CPUOperator):
    """CPU-only variant that delegates to a remote Nemotron Parse endpoint.

    Defaults to the build.nvidia.com chat completions endpoint.
    No local GPU model is loaded.
    """

    DEFAULT_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        nemotron_parse_model: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._invoke_url = (
            str(nemotron_parse_invoke_url or "").strip() or str(invoke_url or "").strip() or self.DEFAULT_INVOKE_URL
        )
        self._nemotron_parse_model = nemotron_parse_model
        validate_nemotron_parse_endpoint_list(self._invoke_url)
        self._api_key = api_key
        self._request_timeout_s = float(request_timeout_s)
        self._task_prompt = str(task_prompt)
        self._remote_retry = RemoteRetryParams(
            remote_max_pool_workers=int(remote_max_pool_workers),
            remote_max_retries=int(remote_max_retries),
            remote_max_429_retries=int(remote_max_429_retries),
        )
        self._model = None
        self._nim_client = NIMClient(max_pool_workers=int(remote_max_pool_workers))
        self._extract_text = bool(extract_text)
        self._extract_tables = bool(extract_tables)
        self._extract_charts = bool(extract_charts)
        self._extract_infographics = bool(extract_infographics)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return nemotron_parse_pages(
            data,
            model=self._model,
            invoke_url=self._invoke_url,
            nemotron_parse_model=self._nemotron_parse_model,
            api_key=self._api_key,
            request_timeout_s=self._request_timeout_s,
            task_prompt=self._task_prompt,
            extract_text=self._extract_text,
            extract_tables=self._extract_tables,
            extract_charts=self._extract_charts,
            extract_infographics=self._extract_infographics,
            remote_retry=self._remote_retry,
            nim_client=self._nim_client,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="nemotron_parse_actor_call", exc=e)
                n = len(out.index)
                out["table"] = [[] for _ in range(n)]
                out["chart"] = [[] for _ in range(n)]
                out["infographic"] = [[] for _ in range(n)]
                out["table_parse"] = [[] for _ in range(n)]
                out["chart_parse"] = [[] for _ in range(n)]
                out["infographic_parse"] = [[] for _ in range(n)]
                out["nemotron_parse_v1_2"] = [payload for _ in range(n)]
                return out
            return [{"nemotron_parse_v1_2": _error_payload(stage="nemotron_parse_actor_call", exc=e)}]


class NemotronParseActor(ArchetypeOperator):
    """Graph-facing Nemotron Parse archetype."""

    _cpu_variant_class = NemotronParseCPUActor
    _gpu_variant_class = NemotronParseGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return bool(
            str(kwargs.get("nemotron_parse_invoke_url") or "").strip() or str(kwargs.get("invoke_url") or "").strip()
        )

    def __init__(
        self,
        *,
        extract_text: bool = False,
        extract_tables: bool = False,
        extract_charts: bool = False,
        extract_infographics: bool = False,
        nemotron_parse_invoke_url: Optional[str] = None,
        nemotron_parse_model: Optional[str] = None,
        invoke_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_s: float = 120.0,
        task_prompt: str = NEMOTRON_PARSE_DEFAULT_TASK_PROMPT,
        remote_max_pool_workers: int = 16,
        remote_max_retries: int = 10,
        remote_max_429_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            nemotron_parse_invoke_url=nemotron_parse_invoke_url,
            nemotron_parse_model=nemotron_parse_model,
            invoke_url=invoke_url,
            api_key=api_key,
            request_timeout_s=request_timeout_s,
            task_prompt=task_prompt,
            remote_max_pool_workers=remote_max_pool_workers,
            remote_max_retries=remote_max_retries,
            remote_max_429_retries=remote_max_429_retries,
            **kwargs,
        )
