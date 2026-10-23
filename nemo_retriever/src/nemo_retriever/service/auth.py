# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bearer authentication and token-to-workspace scope authorization."""

from __future__ import annotations

import hmac
import json
import logging
from pathlib import Path
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from nemo_retriever.service.config import AuthConfig

logger = logging.getLogger(__name__)


_BEARER_PREFIX = "bearer "
_INTERNAL_TOKEN_HEADER = "X-NRL-Internal-Token"
_GATEWAY_HANDOFF_HEADER = "X-NRL-Gateway-Handoff"
_AUTHORIZED_SCOPE_HEADER = "X-NRL-Authorized-Scope"
_CALLER_FINGERPRINT_HEADER = "X-NRL-Caller-Fingerprint"
_GATEWAY_HANDOFF_VALUE = "v1"

_GATEWAY_PROXY_PATHS = frozenset({"/v1/ingest/sidecar", "/v1/ingest/pipeline-config"})


def _is_gateway_proxy_path(path: str) -> bool:
    """Return whether *path* is a gateway-to-worker sidecar or config route."""
    return path in _GATEWAY_PROXY_PATHS or path.startswith("/v1/ingest/sidecar/")


def tokens_match(supplied: str, expected: str) -> bool:
    """Compare two credentials in constant time, whatever characters they hold.

    ``hmac.compare_digest`` raises ``TypeError`` for a ``str`` with non-ASCII
    characters, and Starlette decodes header bytes as latin-1, so comparing the
    header text directly let a client turn a wrong credential into a 500.
    """
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _strip_bearer(value: str) -> str:
    if value.lower().startswith(_BEARER_PREFIX):
        return value[len(_BEARER_PREFIX) :].strip()
    return value.strip()


def auth_headers(config: AuthConfig) -> dict[str, str]:
    """Build the configured service-auth header for internal HTTP calls."""
    token = (config.api_token or "").strip()
    if not token:
        return {}
    value = f"Bearer {token}" if config.header_name.lower() == "authorization" else token
    return {config.header_name: value}


def internal_auth_headers(token: str | None) -> dict[str, str]:
    """Build headers for service-internal calls protected by the shared token."""
    token = (token or "").strip()
    if not token:
        return {}
    return {"X-NRL-Internal-Token": token}


def authorized_scope(request: Request) -> str:
    """Return the middleware-authorized scope; never trust a raw header here."""
    return str(getattr(request.state, "authorized_scope", "default"))


def caller_fingerprint(request: Request) -> str | None:
    """Return the authenticated caller fingerprint, or ``None`` when unavailable.

    The returned value is derived from the bearer token and never contains the
    raw bearer token.
    """
    value = getattr(request.state, "caller_fingerprint", None)
    return str(value) if value else None


def gateway_handoff_headers(
    *, internal_api_token: str | None, scope: str, caller_fingerprint: str | None
) -> dict[str, str]:
    """Build gateway-controlled worker authentication headers.

    Args:
        internal_api_token: Shared credential used to authenticate the worker.
        scope: Scope previously authorized by the gateway.
        caller_fingerprint: Optional non-secret caller identity fingerprint.

    Returns:
        Headers for a restricted gateway-to-worker handoff, or an empty
        dictionary when no internal credential is configured.
    """
    headers = internal_auth_headers(internal_api_token)
    if not headers:
        return {}
    headers[_GATEWAY_HANDOFF_HEADER] = _GATEWAY_HANDOFF_VALUE
    headers[_AUTHORIZED_SCOPE_HEADER] = scope
    if caller_fingerprint:
        headers[_CALLER_FINGERPRINT_HEADER] = caller_fingerprint
    return headers


class ScopeAuthorizer:
    """Resolve credentials to allowed logical scopes without logging secrets."""

    def __init__(self, config: AuthConfig) -> None:
        self.default_scope = config.default_scope.strip() or "default"
        self.enabled = config.enabled
        self.allow_unscoped_dev = config.allow_unscoped_dev
        self._records: list[tuple[str, frozenset[str]]] = []
        if token := (config.api_token or "").strip():
            self._records.append((token, frozenset({self.default_scope})))
        if config.scope_token_file:
            self._load_file(config.scope_token_file)

    def _load_file(self, path: str) -> None:
        try:
            payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to load scope-token secret file {path!r}: {exc}") from exc
        records = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError("scope-token secret file must contain a 'tokens' list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("each scope-token record must be an object")
            token = str(record.get("token") or "").strip()
            scopes = record.get("scopes")
            if not token or not isinstance(scopes, list) or not scopes:
                raise ValueError("scope-token records require a non-empty token and scopes list")
            self._records.append(
                (
                    token,
                    frozenset(str(scope).strip() for scope in scopes if str(scope).strip()),
                )
            )

    def authorize(self, provided_token: str, requested_scope: str | None) -> tuple[str | None, int | None]:
        """Resolve an authorized scope without revealing whether a token was recognized."""

        requested = (requested_scope or self.default_scope).strip() or self.default_scope
        if not self.enabled:
            return requested, None
        if not self._records:
            if self.allow_unscoped_dev:
                return requested, None
            return None, 401
        allowed: frozenset[str] | None = None
        for token, scopes in self._records:
            if tokens_match(provided_token, token):
                allowed = scopes
        if allowed is None:
            return None, 401
        if requested not in allowed:
            return None, 401
        return requested, None


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate public requests and isolate pod-only callback routes."""

    def __init__(
        self,
        app,
        *,
        config: AuthConfig,
        internal_api_token: str | None = None,
        service_mode: str = "standalone",
    ) -> None:
        super().__init__(app)
        self._header = config.header_name
        self._bypass = tuple(config.bypass_paths)
        self._authorizer = ScopeAuthorizer(config)
        self._internal_api_token = (internal_api_token or "").strip()
        self._service_mode = service_mode

    def _set_public_identity(self, request: Request, provided_token: str, scope: str) -> None:
        request.state.authorized_scope = scope
        request.state.caller_fingerprint = (
            hmac.new(self._internal_api_token.encode("utf-8"), provided_token.encode("utf-8"), "sha256").hexdigest()
            if self._internal_api_token and provided_token
            else None
        )

    async def dispatch(self, request: Request, call_next):
        """Authenticate the request and attach its authorized logical scope."""

        path = request.url.path
        if any(path == p or path.startswith(p.rstrip("/") + "/") for p in self._bypass):
            request.state.authorized_scope = self._authorizer.default_scope
            request.state.caller_fingerprint = None
            return await call_next(request)

        if path.startswith("/v1/internal/") and self._internal_api_token:
            supplied = request.headers.get(_INTERNAL_TOKEN_HEADER, "").strip()
            if not supplied or not tokens_match(supplied, self._internal_api_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid internal credential."},
                )
            request.state.authorized_scope = self._authorizer.default_scope
            request.state.caller_fingerprint = None
            return await call_next(request)

        if (
            self._service_mode in ("realtime", "batch")
            and _is_gateway_proxy_path(path)
            and self._internal_api_token
            and request.headers.get(_GATEWAY_HANDOFF_HEADER) == _GATEWAY_HANDOFF_VALUE
        ):
            supplied = request.headers.get(_INTERNAL_TOKEN_HEADER, "").strip()
            scope = request.headers.get(_AUTHORIZED_SCOPE_HEADER, "").strip()
            if supplied and scope and tokens_match(supplied, self._internal_api_token):
                request.state.authorized_scope = scope
                request.state.caller_fingerprint = request.headers.get(_CALLER_FINGERPRINT_HEADER, "").strip() or None
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid internal credential."})

        provided = request.headers.get(self._header, "")
        provided_token = _strip_bearer(provided)
        scope, failure = self._authorizer.authorize(provided_token, request.headers.get("X-NRL-Scope"))
        if failure is not None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        self._set_public_identity(request, provided_token, scope)
        return await call_next(request)
