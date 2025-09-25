import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from nv_ingest_api.util.service_clients.redis.redis_client import RedisClient


logger = logging.getLogger(__name__)

_init_lock: threading.Lock = threading.Lock()
_client: Optional[RedisClient] = None


def _get_client() -> RedisClient:
    global _client
    if _client is not None:
        return _client
    with _init_lock:
        if _client is not None:
            return _client

        host: str = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
        port: int = int(os.getenv("MESSAGE_CLIENT_PORT", "6379"))
        use_ssl: bool = os.getenv("REDIS_USE_SSL", "false").lower() == "true"
        max_pool_size: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "64"))
        max_retries: int = int(os.getenv("REDIS_MAX_RETRIES", "3"))
        max_backoff: int = int(os.getenv("REDIS_MAX_BACKOFF", "32"))
        connection_timeout: int = int(os.getenv("REDIS_CONNECTION_TIMEOUT", "300"))
        message_ttl_seconds_env: Optional[str] = os.getenv("JOB_EVENT_TTL_SECONDS")
        message_ttl_seconds: Optional[int] = int(message_ttl_seconds_env) if message_ttl_seconds_env else None

        _client = RedisClient(
            host=host,
            port=port,
            max_pool_size=max_pool_size,
            use_ssl=use_ssl,
            max_retries=max_retries,
            max_backoff=max_backoff,
            connection_timeout=connection_timeout,
            message_ttl_seconds=message_ttl_seconds,
        )
        logger.debug(
            "Initialized RedisClient for event broadcasting to %s:%d (ssl=%s)",
            host,
            port,
            use_ssl,
        )
        return _client


def broadcast_event(job_id: str, payload: Dict[str, Any]) -> None:
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("job_id must be a non-empty string")
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict serializable to JSON")

    stream_name: str = os.getenv("REDIS_EVENTS_STREAM", "job_events")
    maxlen_env: Optional[str] = os.getenv("REDIS_EVENTS_STREAM_MAXLEN", "10000")
    try:
        maxlen: int = int(maxlen_env) if maxlen_env is not None else 10000
    except Exception:
        maxlen = 10000

    try:
        payload_str: str = json.dumps(payload, separators=(",", ":"))
    except Exception as e:
        raise TypeError(f"payload is not JSON serializable: {e}") from e

    # Use the underlying redis client for XADD (streams)
    redis_client = _get_client().get_client()
    fields: Dict[str, Any] = {"job_id": job_id, "payload": payload_str}
    # Trim approximately to keep memory bounded
    redis_client.xadd(stream_name, fields, maxlen=maxlen, approximate=True)


__all__ = ["broadcast_event"]
