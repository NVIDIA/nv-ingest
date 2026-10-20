# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-side trust boundary for per-request pipeline overrides.

The :class:`~nemo_retriever.service.models.pipeline_spec.PipelineSpec`
shipped by a client cannot be applied blindly: it would let a tenant
redirect NIM traffic, exfiltrate to arbitrary HTTP endpoints, or write
files to attacker-chosen paths. This module centralises that
trust-boundary logic so the router, worker, and ``pipeline-config``
introspection endpoint all see the same picture of what is allowed.

Two layers:

1. **Denylist of endpoint / api_key fields**. The server *always* owns
   these — the only way to influence them is to update
   ``ServiceConfig.nim_endpoints``.
2. **Per-stage allowlist of param keys**. Defaults to a conservative set
   of "shape" fields (chunk sizes, batch sizes, output flags). Operators
   can widen the allowlist via the new ``pipeline_overrides`` section of
   ``retriever-service.yaml``.

The module is pure-Python: it takes plain dicts (the wire format) and
returns plain dicts (the validated, server-merged form). No FastAPI
imports, so unit tests can exercise it directly.
"""

from __future__ import annotations

from typing import Any

from nemo_retriever.common.schemas.pipeline_spec import PipelineSpec


# ----------------------------------------------------------------------
# Trust-sensitive field names — never accept these from a client
# ----------------------------------------------------------------------

_DENYLIST_KEY_SUBSTRINGS: tuple[str, ...] = (
    "invoke_url",
    "endpoint",
    "endpoint_url",
    "api_key",
    "api_base",
    "auth_token",
    "function_id",
    "audio_endpoints",
    "remote_invoke",
    "callback_url",
    "lancedb_uri",
    "storage_uri",
    "vectordb_url",
    "embed_invoke_url",
    "embedding_endpoint",
    "page_elements_invoke_url",
    "ocr_invoke_url",
    "table_structure_invoke_url",
    "nemotron_parse_invoke_url",
    "nemotron_parse_model",
    "profile_name",
)


def _is_trust_sensitive(key: str) -> bool:
    """Return ``True`` when ``key`` matches a denied substring."""
    lower = key.lower()
    return any(needle in lower for needle in _DENYLIST_KEY_SUBSTRINGS)


# ----------------------------------------------------------------------
# Default per-stage allowlist — overridable from YAML
# ----------------------------------------------------------------------

_DEFAULT_ALLOWED_EXTRACT_KEYS: frozenset[str] = frozenset(
    {
        "extract_text",
        "extract_images",
        "extract_nested_images",
        "extract_tables",
        "extract_charts",
        "extract_infographics",
        "extract_page_as_image",
        "method",
        "use_page_elements",
        "use_table_structure",
        "table_output_format",
        "dpi",
        "image_format",
        "jpeg_quality",
        "render_mode",
        "inference_batch_size",
        "ocr_version",
        "output_column",
        "num_detections_column",
        "counts_by_label_column",
    }
)

_DEFAULT_ALLOWED_EMBED_KEYS: frozenset[str] = frozenset(
    {
        "input_type",
        "embed_modality",
        "embed_granularity",
        "text_elements_modality",
        "structured_elements_modality",
        "text_column",
        "inference_batch_size",
        "output_column",
        "embed_inference_batch_size",
        "query_max_length",
        "dimensions",
    }
)

_DEFAULT_ALLOWED_DEDUP_KEYS: frozenset[str] = frozenset({"content_hash", "bbox_iou", "iou_threshold"})

_DEFAULT_ALLOWED_SPLIT_KEYS: frozenset[str] = frozenset(
    {"max_tokens", "overlap_tokens", "tokenizer_model_id", "encoding"}
)

# Sink-stage allowlists (Phase 2). The URI / URL fields in these sets
# are admitted past the allowlist check but then re-validated against
# :class:`SinkUrlAllowlist` so we can mix exact-match and scheme/prefix
# matching for the egress destinations.
_DEFAULT_ALLOWED_STORE_KEYS: frozenset[str] = frozenset(
    {
        "storage_uri",
        # ``storage_options`` omitted: nested fsspec/boto keys can redirect
        # cloud API traffic (SSRF). Operators may re-enable via
        # ``extra_store_keys``; inner keys are then gated by
        # ``_DEFAULT_ALLOWED_STORAGE_OPTIONS_KEYS``.
        "image_format",
        "strip_base64",
    }
)
# Safe inner keys for ``store_params.storage_options`` when an operator
# explicitly widens the top-level allowlist. Empty by default.
_DEFAULT_ALLOWED_STORAGE_OPTIONS_KEYS: frozenset[str] = frozenset()
_DEFAULT_ALLOWED_WEBHOOK_KEYS: frozenset[str] = frozenset(
    {
        "endpoint_url",
        "columns",
        "headers",
        "timeout_s",
        "max_retries",
    }
)
# Inner keys permitted inside ``vdb_upload_params.vdb_kwargs``. The top-level
# allowlist admits ``vdb_kwargs`` as one opaque key; these are the only
# LanceDB constructor / ingest knobs clients may set in service run_mode.
_DEFAULT_ALLOWED_VDB_KWARGS_KEYS: frozenset[str] = frozenset(
    {
        "lancedb_uri",
        "uri",
        "table_name",
        "overwrite",
        "index_type",
        "metric",
        "num_partitions",
        "num_sub_vectors",
        "hybrid",
        "fts_language",
        "vector_dim",
        "on_bad_vectors",
        "fill_value",
        "validate_vector_length",
        "build_index",
        "create_index",
        "nprobes",
        "refine_factor",
        "where",
        "_filter",
    }
)

_DEFAULT_ALLOWED_VDB_UPLOAD_KEYS: frozenset[str] = frozenset(
    {
        "vdb_op",
        "vdb_kwargs",
        # Sidecar metadata (Phase 6) — the caller pre-uploads the
        # dataframe via POST /v1/ingest/sidecar and references the
        # returned opaque id here. ``meta_dataframe`` (the path /
        # in-memory variant) is still rejected — only the id form
        # works in service run_mode.
        "meta_dataframe_id",
        "meta_source_field",
        "meta_fields",
        "meta_join_key",
    }
)

# Caption-stage allowlist (Phase 4). The remote endpoint + API key +
# model name are always server-owned (denylist below); these are the
# behavioural knobs we let callers tune.
_DEFAULT_ALLOWED_CAPTION_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "system_prompt",
        "batch_size",
        "context_text_max_chars",
        "caption_infographics",
        # LLMInferenceParams generic knobs that travel down to the inference call.
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
    }
)

# Keys we forbid even with mode='allow_all'. These configure the
# *local* execution path (in-process GPU) which has no analog when the
# worker is a CPU pod forwarding to a remote NIM. Accepting them would
# silently mislead callers — better to fail fast with a clear message.
_CAPTION_FORBIDDEN_LOCAL_EXECUTION_KEYS: frozenset[str] = frozenset(
    {"device", "hf_cache_dir", "tensor_parallel_size", "gpu_memory_utilization"}
)

# Phase 1 baseline (no sinks). Phase 2 extends this set when at least
# one sink type is enabled in retriever-service.yaml. Phase 4 adds
# ``caption`` once an operator has configured a remote VLM endpoint.
_PHASE_1_STAGES: frozenset[str] = frozenset({"extract", "dedup", "embed", "filter"})
_SINK_STAGES: frozenset[str] = frozenset({"store", "webhook"})  # vdb_upload is a sink but not in stage_order
_CAPTION_STAGE: frozenset[str] = frozenset({"caption"})


class SinkUrlAllowlist:
    """Per-sink-type URL/URI allowlist enforcement.

    Each sink (``store``, ``webhook``, ``vdb_upload``) has its own
    allowlist because the threat profiles differ:

    * ``store`` is restricted to *URI schemes* (``s3://``, ``gs://``, …)
      — wildcards would match the wrong bucket.
    * ``webhook`` is restricted to *URL prefixes* (``https://hooks.example.com/``)
      so operators can pin notifications to a specific tenant domain.
    * ``vdb_upload`` is restricted to *URI schemes* with the same logic
      as ``store`` (LanceDB on cloud storage).

    Empty allowlists mean "the sink is disabled" — the policy rejects
    any client spec that sets the corresponding params block. An entry
    of ``"*"`` opens the gate fully (intended for dev clusters only).
    """

    def __init__(
        self,
        *,
        storage_uri_schemes: list[str] | None = None,
        webhook_url_prefixes: list[str] | None = None,
        vdb_uri_schemes: list[str] | None = None,
    ) -> None:
        self.storage_uri_schemes = list(storage_uri_schemes or [])
        self.webhook_url_prefixes = list(webhook_url_prefixes or [])
        self.vdb_uri_schemes = list(vdb_uri_schemes or [])

    # ── per-sink admission checks ──────────────────────────────────

    def check_storage_uri(self, uri: str | None) -> None:
        if uri is None:
            raise PolicyError(
                "store_params.storage_uri is required in service run_mode "
                "(no local fallback). Set it to a cloud URI such as 's3://bucket/prefix/'.",
                status_code=400,
            )
        if not self.storage_uri_schemes:
            raise PolicyError(
                "store_params: storage sinks are disabled on this service. "
                "Operator must populate sinks.storage_uri_schemes in retriever-service.yaml.",
                status_code=403,
            )
        if "*" in self.storage_uri_schemes:
            return
        scheme = _scheme_of(uri)
        if scheme not in self.storage_uri_schemes:
            raise PolicyError(
                f"store_params.storage_uri scheme {scheme!r} is not in the "
                f"allowed list {sorted(self.storage_uri_schemes)!r}.",
                status_code=403,
            )

    def check_webhook_url(self, url: str | None) -> None:
        if url is None:
            # WebhookParams permits None (the stage becomes a no-op) — but
            # in service mode that's almost certainly a mistake. Reject.
            raise PolicyError(
                "webhook_params.endpoint_url is required in service run_mode.",
                status_code=400,
            )
        if not self.webhook_url_prefixes:
            raise PolicyError(
                "webhook_params: webhooks are disabled on this service. "
                "Operator must populate sinks.webhook_url_prefixes in retriever-service.yaml.",
                status_code=403,
            )
        if "*" in self.webhook_url_prefixes:
            return
        if not any(url.startswith(prefix) for prefix in self.webhook_url_prefixes):
            raise PolicyError(
                f"webhook_params.endpoint_url {url!r} does not match any "
                f"allowed prefix in {self.webhook_url_prefixes!r}.",
                status_code=403,
            )

    def check_vdb_upload(self, params: dict[str, Any] | None) -> None:
        if params is None:
            return
        if not self.vdb_uri_schemes:
            raise PolicyError(
                "vdb_upload_params: per-request vector-DB overrides are disabled. "
                "Operator must populate sinks.vdb_uri_schemes in retriever-service.yaml.",
                status_code=403,
            )
        vdb_kwargs = params.get("vdb_kwargs") or {}
        uri = vdb_kwargs.get("lancedb_uri") or vdb_kwargs.get("uri")
        if uri is None:
            raise PolicyError(
                "vdb_upload_params.vdb_kwargs must include a remote 'lancedb_uri' "
                "(local LanceDB paths are not reachable from worker pods).",
                status_code=400,
            )
        if "*" in self.vdb_uri_schemes:
            return
        scheme = _scheme_of(uri)
        if scheme not in self.vdb_uri_schemes:
            raise PolicyError(
                f"vdb_upload_params.vdb_kwargs.lancedb_uri scheme {scheme!r} is not "
                f"in the allowed list {sorted(self.vdb_uri_schemes)!r}.",
                status_code=403,
            )

    # ── introspection ──────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        return {
            "storage_uri_schemes": self.storage_uri_schemes,
            "webhook_url_prefixes": self.webhook_url_prefixes,
            "vdb_uri_schemes": self.vdb_uri_schemes,
        }

    def has_any_sink_enabled(self) -> bool:
        return bool(self.storage_uri_schemes or self.webhook_url_prefixes or self.vdb_uri_schemes)


def _scheme_of(uri: str) -> str:
    """Return the URI scheme for *uri*, or ``''`` for a bare path."""
    if "://" not in uri:
        return ""
    return uri.split("://", 1)[0].lower() + "://"


class PolicyError(ValueError):
    """Raised by :func:`validate_pipeline_spec` when a client spec is rejected.

    Carries an HTTP-friendly *status_code* (400 for malformed input, 403
    for a policy denial) and a human-readable *detail* string suitable
    for the FastAPI exception handler.
    """

    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ----------------------------------------------------------------------
# Public configuration object — mirrors ServiceConfig.pipeline_overrides
# ----------------------------------------------------------------------


class PipelineOverridesPolicy:
    """Operator-tunable view of what client overrides are admissible.

    Operators set this via ``pipeline_overrides`` in ``retriever-service.yaml``;
    callers in the router/worker get a single :class:`PipelineOverridesPolicy`
    instance from ``ServiceConfig`` and pass it to :func:`validate_pipeline_spec`.
    """

    def __init__(
        self,
        *,
        mode: str = "allow_list",
        allowed_stages: frozenset[str] | None = None,
        extra_extract_keys: frozenset[str] = frozenset(),
        extra_embed_keys: frozenset[str] = frozenset(),
        extra_dedup_keys: frozenset[str] = frozenset(),
        extra_split_keys: frozenset[str] = frozenset(),
        extra_store_keys: frozenset[str] = frozenset(),
        extra_storage_options_keys: frozenset[str] = frozenset(),
        extra_webhook_keys: frozenset[str] = frozenset(),
        extra_vdb_upload_keys: frozenset[str] = frozenset(),
        extra_vdb_kwargs_keys: frozenset[str] = frozenset(),
        extra_caption_keys: frozenset[str] = frozenset(),
        sinks: SinkUrlAllowlist | None = None,
        caption_enabled: bool = False,
    ) -> None:
        if mode not in {"reject", "allow_list", "allow_all"}:
            raise ValueError(
                f"pipeline_overrides.mode must be one of 'reject', 'allow_list', 'allow_all'; got {mode!r}"
            )
        self.mode = mode
        self.sinks = sinks or SinkUrlAllowlist()
        self.caption_enabled = caption_enabled
        # The base stage set grows incrementally: sinks open up
        # ``store``/``webhook``; a configured caption endpoint opens up
        # ``caption``.
        base_stages = _PHASE_1_STAGES
        if self.sinks.has_any_sink_enabled():
            base_stages = base_stages | _SINK_STAGES
        if self.caption_enabled:
            base_stages = base_stages | _CAPTION_STAGE
        self.allowed_stages = allowed_stages if allowed_stages is not None else base_stages
        self.allowed_extract_keys = _DEFAULT_ALLOWED_EXTRACT_KEYS | extra_extract_keys
        self.allowed_embed_keys = _DEFAULT_ALLOWED_EMBED_KEYS | extra_embed_keys
        self.allowed_dedup_keys = _DEFAULT_ALLOWED_DEDUP_KEYS | extra_dedup_keys
        self.allowed_split_keys = _DEFAULT_ALLOWED_SPLIT_KEYS | extra_split_keys
        self.allowed_store_keys = _DEFAULT_ALLOWED_STORE_KEYS | extra_store_keys
        self.allowed_storage_options_keys = _DEFAULT_ALLOWED_STORAGE_OPTIONS_KEYS | extra_storage_options_keys
        self.allowed_webhook_keys = _DEFAULT_ALLOWED_WEBHOOK_KEYS | extra_webhook_keys
        self.allowed_vdb_upload_keys = _DEFAULT_ALLOWED_VDB_UPLOAD_KEYS | extra_vdb_upload_keys
        self.allowed_vdb_kwargs_keys = _DEFAULT_ALLOWED_VDB_KWARGS_KEYS | extra_vdb_kwargs_keys
        self.allowed_caption_keys = _DEFAULT_ALLOWED_CAPTION_KEYS | extra_caption_keys

    def describe(self) -> dict[str, Any]:
        """Render the policy as a JSON-safe dict for the introspection endpoint."""
        return {
            "mode": self.mode,
            "allowed_stages": sorted(self.allowed_stages),
            "allowed_extract_keys": sorted(self.allowed_extract_keys),
            "allowed_embed_keys": sorted(self.allowed_embed_keys),
            "allowed_dedup_keys": sorted(self.allowed_dedup_keys),
            "allowed_split_keys": sorted(self.allowed_split_keys),
            "allowed_store_keys": sorted(self.allowed_store_keys),
            "allowed_storage_options_keys": sorted(self.allowed_storage_options_keys),
            "allowed_webhook_keys": sorted(self.allowed_webhook_keys),
            "allowed_vdb_upload_keys": sorted(self.allowed_vdb_upload_keys),
            "allowed_vdb_kwargs_keys": sorted(self.allowed_vdb_kwargs_keys),
            "allowed_caption_keys": sorted(self.allowed_caption_keys),
            "caption_enabled": self.caption_enabled,
            "denied_key_substrings": sorted(_DENYLIST_KEY_SUBSTRINGS),
            "sinks": self.sinks.describe(),
        }


# ----------------------------------------------------------------------
# Validation entry point
# ----------------------------------------------------------------------


def _scrub_trust_sensitive(params: dict[str, Any] | None, stage: str) -> dict[str, Any] | None:
    """Strip trust-sensitive keys from a params dict, raising on hit.

    We *reject* (rather than silently drop) so the client gets a clear
    error and knows their request will not behave as written.
    """
    if params is None:
        return None
    bad = [k for k in params if _is_trust_sensitive(k)]
    if bad:
        raise PolicyError(
            f"{stage}_params: rejected trust-sensitive keys {bad!r}. "
            "Endpoint URLs and API keys are configured via the server's "
            "nim_endpoints section and cannot be overridden per-request.",
            status_code=403,
        )
    return params


def _scrub_trust_sensitive_except(
    params: dict[str, Any] | None,
    stage: str,
    *,
    allow: set[str],
) -> dict[str, Any] | None:
    """Variant of :func:`_scrub_trust_sensitive` that permits *allow* keys.

    Sink stages legitimately need *one* trust-sensitive field — the sink
    destination itself. The sink URL allowlist (``SinkUrlAllowlist``)
    then validates that field separately. Everything else in the
    denylist still rejects the request.
    """
    if params is None:
        return None
    bad = [k for k in params if _is_trust_sensitive(k) and k not in allow]
    if bad:
        raise PolicyError(
            f"{stage}_params: rejected trust-sensitive keys {bad!r}. "
            "Endpoint URLs and API keys are configured via the server's "
            "nim_endpoints section and cannot be overridden per-request.",
            status_code=403,
        )
    return params


def _scrub_nested_trust_sensitive(
    parent: dict[str, Any] | None,
    nested_key: str,
    *,
    path: str,
    trust_allow: set[str] | None = None,
) -> None:
    """Reject trust-sensitive keys inside a nested mapping (one level deep)."""
    if parent is None:
        return
    nested = parent.get(nested_key)
    if nested is None:
        return
    if not isinstance(nested, dict):
        raise PolicyError(f"{path} must be a mapping", status_code=400)
    bad = [k for k in nested if _is_trust_sensitive(k) and (trust_allow is None or k not in trust_allow)]
    if bad:
        raise PolicyError(
            f"{path}: rejected trust-sensitive keys {bad!r}. "
            "Endpoint URLs, credentials, and storage client overrides are "
            "not accepted inside nested sink kwargs.",
            status_code=403,
        )


def _enforce_storage_options_locked(
    store_params: dict[str, Any],
    policy: PipelineOverridesPolicy,
) -> None:
    """Trust denylist + inner allowlist for ``storage_options`` (always enforced)."""
    path = "store_params.storage_options"
    _scrub_nested_trust_sensitive(store_params, "storage_options", path=path)
    nested = store_params.get("storage_options")
    if not isinstance(nested, dict):
        raise PolicyError(f"{path} must be a mapping", status_code=400)
    extras = [k for k in nested if k not in policy.allowed_storage_options_keys]
    if extras:
        raise PolicyError(
            f"{path}: keys {extras!r} are not in the allow_list. "
            "Per-request storage client overrides are disabled by default; "
            "ask the operator to widen pipeline_overrides.extra_storage_options_keys.",
            status_code=403,
        )


def _enforce_nested_allowlist(
    parent: dict[str, Any] | None,
    nested_key: str,
    *,
    path: str,
    allowed: frozenset[str],
    mode: str,
    trust_allow: set[str] | None = None,
) -> None:
    """Trust denylist plus inner-key allowlist (skipped when ``mode='allow_all'``)."""
    _scrub_nested_trust_sensitive(parent, nested_key, path=path, trust_allow=trust_allow)
    if parent is None or mode == "allow_all":
        return
    nested = parent.get(nested_key)
    if not isinstance(nested, dict):
        return
    extras = [k for k in nested if k not in allowed]
    if extras:
        raise PolicyError(
            f"{path}: keys {extras!r} are not in the allow_list. "
            "Ask the service operator to widen pipeline_overrides.extra_vdb_kwargs_keys "
            "or remove them from your request.",
            status_code=403,
        )


def _enforce_allowlist(
    params: dict[str, Any] | None,
    allowed: frozenset[str],
    stage: str,
    *,
    mode: str,
) -> dict[str, Any] | None:
    """Apply per-stage allowlist enforcement based on the policy mode."""
    if params is None or mode == "allow_all":
        return params
    extras = [k for k in params if k not in allowed]
    if not extras:
        return params
    if mode == "reject":
        raise PolicyError(
            f"{stage}_params: client overrides are disabled (pipeline_overrides.mode='reject'). "
            f"Offending keys: {extras!r}.",
            status_code=403,
        )
    raise PolicyError(
        f"{stage}_params: keys {extras!r} are not in the allow_list. "
        "Ask the service operator to widen pipeline_overrides.extra_*_keys "
        "or remove them from your request.",
        status_code=403,
    )


def validate_pipeline_spec(
    spec: PipelineSpec | None,
    policy: PipelineOverridesPolicy,
) -> PipelineSpec | None:
    """Return a sanitized copy of *spec* or raise :class:`PolicyError`.

    Returns ``None`` when the spec is missing or empty (so the worker can
    short-circuit to the legacy startup-baked pipeline).
    """
    if spec is None or spec.is_empty():
        return None

    result_schema_only = (
        spec.result_schema != "legacy"
        and spec.extraction_mode in ("pdf", "auto")
        and spec.extract_params is None
        and spec.embed_params is None
        and spec.dedup_params is None
        and spec.caption_params is None
        and spec.store_params is None
        and spec.vdb_upload_params is None
        and spec.webhook_params is None
        and spec.split_config is None
        and spec.pdf_split is None
        and not spec.stage_order
        and not spec.return_embeddings
        and not spec.return_images
    )
    if result_schema_only:
        return spec

    if policy.mode == "reject":
        raise PolicyError(
            "Per-request pipeline overrides are disabled on this service "
            "(pipeline_overrides.mode='reject'). Update retriever-service.yaml "
            "to enable them.",
            status_code=403,
        )

    for stage_name in spec.stage_order:
        if stage_name not in policy.allowed_stages:
            raise PolicyError(
                f"stage {stage_name!r} is not in pipeline_overrides.allowed_stages. "
                f"Allowed in this phase: {sorted(policy.allowed_stages)}.",
                status_code=403,
            )

    # caption_params is admitted only when the operator has configured a
    # remote VLM endpoint. We also reject local-execution keys outright
    # because the CPU worker pod cannot honor them — surfacing the
    # mismatch immediately is friendlier than silently ignoring them.
    if spec.caption_params is not None:
        if not policy.caption_enabled:
            raise PolicyError(
                "caption_params overrides require an operator-configured caption "
                "endpoint. Set caption.endpoint_url in retriever-service.yaml first.",
                status_code=403,
            )
        forbidden = [k for k in spec.caption_params if k in _CAPTION_FORBIDDEN_LOCAL_EXECUTION_KEYS]
        if forbidden:
            raise PolicyError(
                f"caption_params: keys {forbidden!r} configure local in-process GPU "
                "execution and have no effect in service run_mode. Remove them and "
                "rely on the operator-configured caption endpoint.",
                status_code=403,
            )

    # Endpoint/API-key denylist applies to every params block, even ones
    # we currently accept — defense in depth in case a new field name
    # slips into the allowlist without being audited.
    _scrub_trust_sensitive(spec.extract_params, "extract")
    _scrub_trust_sensitive(spec.embed_params, "embed")
    _scrub_trust_sensitive(spec.dedup_params, "dedup")
    _scrub_trust_sensitive(spec.caption_params, "caption")
    _scrub_trust_sensitive(spec.split_config, "split")
    # ``endpoint_url`` IS legitimate in webhook_params (it's the sink
    # destination). Strip the rest of the trust-sensitive surface but
    # leave that one through; the sink allowlist polices it next.
    _scrub_trust_sensitive_except(spec.webhook_params, "webhook", allow={"endpoint_url"})
    # Same idea for store_params (storage_uri) and vdb_upload_params (lancedb_uri).
    _scrub_trust_sensitive_except(spec.store_params, "store", allow={"storage_uri"})
    _scrub_trust_sensitive_except(spec.vdb_upload_params, "vdb_upload", allow={"lancedb_uri"})

    # Reject the local-path variant of sidecar metadata BEFORE the
    # generic allowlist check so callers get a specific pointer at
    # the upload endpoint instead of an "extra key" message.
    if spec.vdb_upload_params is not None and spec.vdb_upload_params.get("meta_dataframe") is not None:
        raise PolicyError(
            "vdb_upload_params.meta_dataframe (path / in-memory DataFrame) is "
            "not reachable from the worker pod. Pre-upload the file via POST "
            "/v1/ingest/sidecar and pass the returned id as meta_dataframe_id.",
            status_code=400,
        )

    _enforce_allowlist(spec.extract_params, policy.allowed_extract_keys, "extract", mode=policy.mode)
    _enforce_allowlist(spec.embed_params, policy.allowed_embed_keys, "embed", mode=policy.mode)
    _enforce_allowlist(spec.dedup_params, policy.allowed_dedup_keys, "dedup", mode=policy.mode)
    _enforce_allowlist(spec.store_params, policy.allowed_store_keys, "store", mode=policy.mode)
    _enforce_allowlist(spec.webhook_params, policy.allowed_webhook_keys, "webhook", mode=policy.mode)
    _enforce_allowlist(spec.vdb_upload_params, policy.allowed_vdb_upload_keys, "vdb_upload", mode=policy.mode)
    if spec.vdb_upload_params is not None:
        _enforce_nested_allowlist(
            spec.vdb_upload_params,
            "vdb_kwargs",
            path="vdb_upload_params.vdb_kwargs",
            allowed=policy.allowed_vdb_kwargs_keys,
            mode=policy.mode,
            trust_allow={"lancedb_uri", "uri"},
        )
    if spec.store_params is not None and spec.store_params.get("storage_options") is not None:
        _enforce_storage_options_locked(spec.store_params, policy)
    _enforce_allowlist(spec.caption_params, policy.allowed_caption_keys, "caption", mode=policy.mode)
    if spec.split_config is not None:
        for source_type, cfg in spec.split_config.items():
            if not isinstance(cfg, dict):
                continue
            _enforce_allowlist(cfg, policy.allowed_split_keys, f"split[{source_type}]", mode=policy.mode)

    # Sink URL allowlists — gatekeep destinations regardless of allowlist mode.
    if spec.store_params is not None:
        policy.sinks.check_storage_uri(spec.store_params.get("storage_uri"))
    if spec.webhook_params is not None:
        policy.sinks.check_webhook_url(spec.webhook_params.get("endpoint_url"))
    if spec.vdb_upload_params is not None:
        policy.sinks.check_vdb_upload(spec.vdb_upload_params)

    return spec
