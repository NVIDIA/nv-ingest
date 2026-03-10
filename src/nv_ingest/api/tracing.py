# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint tracing utilities."""

from functools import wraps
from inspect import iscoroutinefunction
from typing import Any, Callable, Optional, TypeVar

from fastapi import Request, Response
from opentelemetry import trace

F = TypeVar("F", bound=Callable[..., Any])

tracer = trace.get_tracer(__name__)


def traced_endpoint(name: Optional[str] = None) -> Callable[[F], F]:
    """Wrap a FastAPI endpoint with a span whose name defaults to the function name.

    The decorator preserves the wrapped callable's signature so FastAPI can continue
    to perform dependency injection and generate OpenAPI documentation correctly.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("nv_ingest.endpoint", func.__qualname__)
                    _record_http_request(span, args, kwargs)
                    response = await func(*args, **kwargs)
                    _record_http_response(span, response)
                    return response

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("nv_ingest.endpoint", func.__qualname__)
                _record_http_request(span, args, kwargs)
                result = func(*args, **kwargs)
                _record_http_response(span, result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _record_http_request(span, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    request = _find_type(Request, args, kwargs)
    if request is None:
        return
    span.set_attribute("http.method", request.method)
    span.set_attribute("http.url", str(request.url))


def _record_http_response(span, response: Any) -> None:
    maybe_response = response if isinstance(response, Response) else None
    if maybe_response is None:
        maybe_response = _find_type(Response, (response,), {})
    if maybe_response is None:
        return
    span.set_attribute("http.status_code", maybe_response.status_code)


def _find_type(expected_type: type, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[Any]:
    """Return the first argument matching ``expected_type`` from args or kwargs."""

    for arg in args:
        if isinstance(arg, expected_type):
            return arg
    for value in kwargs.values():
        if isinstance(value, expected_type):
            return value
    return None
