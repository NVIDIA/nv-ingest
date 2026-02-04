from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class Metrics:
    """Minimal metrics surface that works in-process and inside Ray tasks."""

    def inc(self, name: str, value: int = 1, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        """Increment a counter."""

    def observe(self, name: str, value: float, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        """Observe a histogram/summary value."""

    @contextlib.contextmanager
    def timeit(self, name: str, *, attrs: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - t0, attrs=attrs)


@dataclass
class NullMetrics(Metrics):
    def inc(self, name: str, value: int = 1, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        return None

    def observe(self, name: str, value: float, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        return None


@dataclass
class LoggingMetrics(Metrics):
    """Emit metrics as structured log lines.

    This is intentionally simple so it can be used in Ray workers without any global
    metrics backend configured yet. You can later replace this with an OpenTelemetry-backed
    implementation.
    """

    prefix: str = "retriever.metric"
    extra: Dict[str, Any] = field(default_factory=dict)

    def inc(self, name: str, value: int = 1, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        payload = {"name": name, "value": value, "type": "counter"}
        payload.update(self.extra)
        if attrs:
            payload["attrs"] = attrs
        logger.info("%s %s", self.prefix, payload)

    def observe(self, name: str, value: float, *, attrs: Optional[Dict[str, Any]] = None) -> None:
        payload = {"name": name, "value": value, "type": "histogram"}
        payload.update(self.extra)
        if attrs:
            payload["attrs"] = attrs
        logger.info("%s %s", self.prefix, payload)


def safe_metrics(metrics: Optional[Metrics]) -> Metrics:
    return metrics if metrics is not None else NullMetrics()

