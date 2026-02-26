# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ASRActor: Ray Data map_batches callable for speech-to-text (Parakeet/Riva gRPC).

Consumes chunk rows (path, bytes, source_path, duration, chunk_index, metadata)
and produces rows with text (transcript) for downstream embed/VDB.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from retriever.params import ASRParams

logger = logging.getLogger(__name__)

# Default NGC/NVCF Parakeet gRPC endpoint when using cloud ASR
DEFAULT_NGC_ASR_GRPC_ENDPOINT = "grpc.nvcf.nvidia.com:443"
# Default NVCF function ID for Parakeet NIM (same as nv-ingest default_libmode_pipeline_impl)
DEFAULT_NGC_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"


def asr_params_from_env(
    *,
    grpc_endpoint_var: str = "AUDIO_GRPC_ENDPOINT",
    auth_token_var: str = "NGC_API_KEY",
    function_id_var: str = "AUDIO_FUNCTION_ID",
    default_grpc_endpoint: Optional[str] = DEFAULT_NGC_ASR_GRPC_ENDPOINT,
    default_function_id: Optional[str] = DEFAULT_NGC_ASR_FUNCTION_ID,
) -> ASRParams:
    """
    Build ASRParams from environment variables for cloud/NGC ASR.

    - AUDIO_GRPC_ENDPOINT: gRPC endpoint (default: grpc.nvcf.nvidia.com:443 for NGC).
    - NGC_API_KEY: Bearer token for NGC/NVCF (required for cloud).
    - AUDIO_FUNCTION_ID: NVCF function ID for the Parakeet NIM (default: same as nv-ingest libmode).

    Returns ASRParams with auth_token and function_id set from env when present.
    When NGC_API_KEY is set but AUDIO_FUNCTION_ID is not, uses the nv-ingest default Parakeet NIM function ID.
    """
    import os

    grpc_endpoint = (os.environ.get(grpc_endpoint_var) or "").strip() or default_grpc_endpoint
    auth_token = (os.environ.get(auth_token_var) or "").strip() or None
    function_id = (os.environ.get(function_id_var) or "").strip() or None
    if auth_token and function_id is None and default_function_id:
        function_id = default_function_id

    return ASRParams(
        audio_endpoints=(grpc_endpoint, None),
        audio_infer_protocol="grpc",
        function_id=function_id,
        auth_token=auth_token,
    )

try:
    from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import (
        create_audio_inference_client,
    )
    _PARAKEET_AVAILABLE = True
except ImportError:
    create_audio_inference_client = None  # type: ignore[misc, assignment]
    _PARAKEET_AVAILABLE = False


def _get_client(params: ASRParams):  # noqa: ANN201
    if not _PARAKEET_AVAILABLE or create_audio_inference_client is None:
        raise RuntimeError(
            "ASRActor requires nv-ingest-api (Parakeet client). "
            "Install with: pip install nv-ingest-api (or add nv-ingest-api to dependencies)."
        )
    grpc_endpoint = (params.audio_endpoints[0] or "").strip() or None
    http_endpoint = (params.audio_endpoints[1] or "").strip() or None
    if not grpc_endpoint:
        raise ValueError(
            "ASR audio_endpoints[0] (gRPC) must be set for Parakeet (e.g. localhost:50051 or grpc.nvcf.nvidia.com:443)."
        )
    return create_audio_inference_client(
        (grpc_endpoint, http_endpoint or ""),
        infer_protocol=params.audio_infer_protocol or "grpc",
        auth_token=params.auth_token,
        function_id=params.function_id,
        use_ssl=bool("nvcf.nvidia.com" in grpc_endpoint and params.function_id),
        ssl_cert=None,
    )


class ASRActor:
    """
    Ray Data map_batches callable: chunk rows (path/bytes) -> rows with text (transcript).

    Uses Parakeet (Riva ASR) via gRPC. Output rows have path, text, page_number, metadata
    for downstream explode_content_to_rows and embed.
    """

    def __init__(self, params: ASRParams | None = None) -> None:
        self._params = params or ASRParams()
        self._client = _get_client(self._params)

    def __call__(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )

        out_rows: List[Dict[str, Any]] = []
        for _, row in batch_df.iterrows():
            try:
                out_row = self._transcribe_one(row)
                if out_row is not None:
                    out_rows.append(out_row)
            except Exception as e:
                logger.exception("ASR failed for row path=%s: %s", row.get("path"), e)
                continue

        if not out_rows:
            return pd.DataFrame(
                columns=["path", "source_path", "duration", "chunk_index", "metadata", "page_number", "text"]
            )
        return pd.DataFrame(out_rows)

    def _transcribe_one(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        raw = row.get("bytes")
        path = row.get("path")
        if raw is None and path:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except Exception as e:
                logger.warning("Could not read %s: %s", path, e)
                return None
        if raw is None:
            return None

        audio_b64 = base64.b64encode(raw).decode("ascii")
        try:
            segments, transcript = self._client.infer(
                audio_b64,
                model_name="parakeet",
            )
        except Exception as e:
            logger.warning("Parakeet infer failed for path=%s: %s", path, e)
            return None

        transcript = transcript if transcript else ""
        source_path = row.get("source_path", path)
        duration = row.get("duration")
        chunk_index = row.get("chunk_index", 0)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {"source_path": source_path, "chunk_index": chunk_index, "duration": duration}
        page_number = row.get("page_number", chunk_index)

        return {
            "path": path,
            "source_path": source_path,
            "duration": duration,
            "chunk_index": chunk_index,
            "metadata": metadata,
            "page_number": page_number,
            "text": transcript,
        }


def apply_asr_to_df(
    batch_df: pd.DataFrame,
    asr_params: Optional[dict] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Inprocess helper: apply ASR to a DataFrame of chunk rows; returns DataFrame with text column set.

    Used by InProcessIngestor when _pipeline_type == "audio". asr_params can be a dict
    to construct ASRParams (e.g. from model_dump()).
    """
    params = ASRParams(**(asr_params or {}))
    actor = ASRActor(params=params)
    return actor(batch_df)
