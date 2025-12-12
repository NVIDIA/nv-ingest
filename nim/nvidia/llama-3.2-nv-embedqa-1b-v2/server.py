from __future__ import annotations

import asyncio
import base64
import ctypes
import ctypes.util
import logging
import mmap
import os
import threading
import time
from concurrent import futures
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import grpc
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from google.protobuf import json_format, text_format
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModel, AutoTokenizer
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc
from tritonclient.utils import deserialize_bytes_tensor

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except Exception:  # pragma: no cover - optional dependency
    trace = None
    otel_metrics = None

logger = logging.getLogger("llama-3.2-nv-embedqa-1b-v2")


TEXT_INPUT_NAME = "TEXT"
DIMENSIONS_INPUT_NAME = "DIMENSIONS"
OUTPUT_NAME = "EMBEDDINGS"

TruncationMode = Literal["NONE", "END", "START"]
EncodingFormat = Literal["float", "base64"]


class ServiceSettings(BaseSettings):
    model_id: str = Field("nvidia/llama-3.2-nv-embedqa-1b-v2", alias="MODEL_ID")
    model_version: str = Field("1.10.0", alias="MODEL_VERSION")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    grpc_port: int = Field(8001, alias="NIM_GRPC_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    max_batch_size_env: Optional[int] = Field(None, alias="NIM_TRITON_MAX_BATCH_SIZE")
    triton_config_path: Optional[str] = Field(None, alias="TRITON_MODEL_CONFIG_PATH")
    max_sequence_length: Optional[int] = Field(None, alias="MAX_SEQUENCE_LENGTH")
    rate_limit: Optional[int] = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    enable_model_control: bool = Field(True, alias="NIM_TRITON_ENABLE_MODEL_CONTROL")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: Optional[str] = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: Optional[str] = Field(None, alias="NIM_OTEL_SERVICE_NAME")
    auth_token: Optional[str] = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: Optional[str] = Field(None, alias="NIM_NGC_API_KEY")
    nvidia_auth_token: Optional[str] = Field(None, alias="NVIDIA_API_KEY")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    @property
    def resolved_auth_tokens(self) -> set[str]:
        return {token for token in (self.auth_token, self.fallback_auth_token, self.nvidia_auth_token) if token}


settings = ServiceSettings()
logging.basicConfig(level=logging.DEBUG if settings.log_verbose else logging.INFO)
logger.setLevel(settings.log_level.upper())


@dataclass
class TritonModelConfig:
    name: str
    version: str
    max_batch_size: int
    embedding_dim: int


@dataclass
class ModelState:
    loaded: bool = False
    inference_count: int = 0
    last_inference_ns: int = 0
    success_ns_total: int = 0


@dataclass
class SystemSharedMemoryRegion:
    name: str
    key: str
    offset: int
    byte_size: int
    shm: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: object | None = None

    @property
    def buffer(self) -> memoryview:
        if self.shm is not None:
            return self.shm.buf
        if self.mmap_obj is not None:
            return memoryview(self.mmap_obj)
        raise RuntimeError("Shared memory region has no buffer")

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
        if self.mmap_obj is not None:
            self.mmap_obj.close()
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to close shared memory file handle", exc_info=True)


@dataclass
class CudaSharedMemoryRegion:
    name: str
    device_id: int
    byte_size: int
    raw_handle: bytes
    offset: int = 0
    pointer: int | None = None


_cudart: ctypes.CDLL | None = None

CUDA_IPC_FLAG = 1
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]
    encoding_format: EncodingFormat = "float"
    input_type: str = "passage"
    truncate: TruncationMode = "END"
    dimensions: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, values: Any) -> Any:
        data = dict(values) if isinstance(values, dict) else values
        if not isinstance(data, dict) or "input" not in data:
            raise ValueError("input is required")
        provided_input = data.get("input")
        if isinstance(provided_input, str):
            data["input"] = [provided_input]
        elif isinstance(provided_input, Sequence):
            data["input"] = list(provided_input)
        else:
            raise ValueError("input must be a string or list of strings")
        return data

    @field_validator("input")
    @classmethod
    def _validate_input(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("input must contain at least one string")
        cleaned: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("input values must be strings")
            text = item.strip()
            if not text:
                raise ValueError("input entries must not be empty")
            cleaned.append(text)
        return cleaned

    @field_validator("input_type")
    @classmethod
    def _normalize_input_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("input_type must not be empty")
        return normalized

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("dimensions must be positive")
        return value


_model_condition = threading.Condition()
_active_inference = 0
_system_shared_memory: Dict[str, SystemSharedMemoryRegion] = {}
_cuda_shared_memory: Dict[str, CudaSharedMemoryRegion] = {}
_shared_memory_lock = threading.Lock()


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _read_max_batch_size() -> int:
    config_text: Optional[str] = None
    if settings.triton_config_path and os.path.exists(settings.triton_config_path):
        try:
            with open(settings.triton_config_path, "r", encoding="utf-8") as handle:
                config_text = handle.read()
            config_proto = model_config_pb2.ModelConfig()
            text_format.Parse(config_text, config_proto)
            if config_proto.max_batch_size > 0:
                return int(config_proto.max_batch_size)
        except Exception as exc:  # pragma: no cover - surfaced to logs
            logger.warning("Failed to parse Triton config at %s: %s", settings.triton_config_path, exc)
        if config_text:
            for line in config_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("max_batch_size"):
                    _, value = stripped.split(":", maxsplit=1)
                    parsed = int(value.strip())
                    if parsed > 0:
                        return parsed
    if settings.max_batch_size_env is not None and settings.max_batch_size_env > 0:
        return settings.max_batch_size_env
    return 30


def _configure_tracing() -> Optional["trace.Tracer"]:
    if not settings.enable_otel or not trace or not otel_metrics:
        return None
    try:
        resource = Resource.create({"service.name": settings.otel_service_name or settings.model_id})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=settings.otel_endpoint or "http://localhost:4318/v1/traces",
                    insecure=True,
                )
            )
        )
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=settings.otel_endpoint or "http://localhost:4318/v1/metrics",
                insecure=True,
            )
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        otel_metrics.set_meter_provider(meter_provider)
        return trace.get_tracer(settings.otel_service_name or settings.model_id)
    except Exception:  # pragma: no cover - optional
        logger.exception("Failed to configure OTEL")
        return None


TRACER = _configure_tracing()


model_state = ModelState()


def _dispose_assets(assets: Optional[Dict[str, Any]]) -> None:
    if not assets:
        return
    model = assets.get("model")
    tokenizer = assets.get("tokenizer")
    try:
        if isinstance(model, torch.nn.Module):
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to dispose model cleanly", exc_info=True)
    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("Failed to dispose tokenizer", exc_info=True)


def _is_model_ready() -> bool:
    with _model_condition:
        assets_loaded = getattr(app.state, "assets", None) is not None
        return model_state.loaded and assets_loaded


def _load_model() -> None:
    with _model_condition:
        if model_state.loaded and getattr(app.state, "assets", None) is not None:
            return
        while _active_inference > 0:
            _model_condition.wait()
    assets = _load_assets()
    max_batch_size = _read_max_batch_size()
    with _model_condition:
        app.state.assets = assets
        model_state.loaded = True
        _model_condition.notify_all()
    if "triton_config" in globals():
        triton_config.max_batch_size = max_batch_size
        triton_config.embedding_dim = assets["embedding_dim"]


def _unload_model() -> None:
    assets: Optional[Dict[str, Any]] = None
    with _model_condition:
        if not model_state.loaded and getattr(app.state, "assets", None) is None:
            return
        while _active_inference > 0:
            _model_condition.wait()
        assets = getattr(app.state, "assets", None)
        app.state.assets = None
        model_state.loaded = False
        _model_condition.notify_all()
    _dispose_assets(assets)


def _record_inference_success(duration_ns: int) -> None:
    with _model_condition:
        model_state.inference_count += 1
        model_state.last_inference_ns = time.time_ns()
        model_state.success_ns_total += duration_ns


@contextmanager
def _model_usage_http():
    global _active_inference
    with _model_condition:
        if not model_state.loaded or getattr(app.state, "assets", None) is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        _active_inference += 1
        assets = app.state.assets
    try:
        yield assets
    finally:
        with _model_condition:
            _active_inference -= 1
            if _active_inference == 0:
                _model_condition.notify_all()


@contextmanager
def _model_usage_grpc(context: grpc.ServicerContext):
    global _active_inference
    with _model_condition:
        if not model_state.loaded or getattr(app.state, "assets", None) is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        _active_inference += 1
        assets = app.state.assets
    try:
        yield assets
    finally:
        with _model_condition:
            _active_inference -= 1
            if _active_inference == 0:
                _model_condition.notify_all()


class AsyncRateLimiter:
    def __init__(self, max_concurrent: Optional[int]):
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self):
        if not self._sem:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


class SyncRateLimiter:
    def __init__(self, max_concurrent: Optional[int]):
        self._sem = threading.BoundedSemaphore(max_concurrent) if max_concurrent else None

    @contextmanager
    def limit(self):
        if not self._sem:
            yield
            return
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


REQUEST_COUNTER = Counter("nim_requests_total", "Total requests served", ["protocol", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "nim_request_latency_seconds",
    "Latency for requests",
    ["protocol", "endpoint"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
)
INFLIGHT_GAUGE = Gauge("nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"])

async_rate_limiter = AsyncRateLimiter(settings.rate_limit)
sync_rate_limiter = SyncRateLimiter(settings.rate_limit)

_metrics_started = False
if settings.metrics_port and not settings.require_auth and not _metrics_started:
    start_http_server(settings.metrics_port)
    _metrics_started = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


def _extract_token(header_value: Optional[str]) -> Optional[str]:
    if header_value is None:
        return None
    stripped = header_value.strip()
    if stripped == "":
        return None
    parts = stripped.split(" ", maxsplit=1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        return token if token else None
    return stripped


def _is_authorized(header_value: Optional[str]) -> bool:
    token = _extract_token(header_value)
    if not settings.require_auth:
        return True
    if token is None:
        return False
    if settings.resolved_auth_tokens:
        return token in settings.resolved_auth_tokens
    return True


async def _require_http_auth(request: Request) -> None:
    header = request.headers.get("authorization") or request.headers.get("ngc-api-key")
    if _is_authorized(header):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def _require_grpc_auth(context: grpc.ServicerContext) -> None:
    metadata: Mapping[str, str] = {k.lower(): v for k, v in context.invocation_metadata()}
    header = metadata.get("authorization") or metadata.get("ngc-api-key")
    if _is_authorized(header):
        return
    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Unauthorized")


def _load_assets() -> Dict[str, Any]:
    device = _select_device()
    dtype = _select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "right"

    model = AutoModel.from_pretrained(
        settings.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    configured_max_length = (
        settings.max_sequence_length
        or getattr(model.config, "max_position_embeddings", None)
        or 8192
    )
    embedding_dim = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "d_model", None)
        or 2048
    )

    logger.info(
        "Loaded %s on %s with dtype %s (max_length=%d, embedding_dim=%d, batch_limit=%d)",
        settings.model_id,
        device,
        dtype,
        configured_max_length,
        embedding_dim,
        _read_max_batch_size(),
    )

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "dtype": dtype,
        "max_length": int(configured_max_length),
        "embedding_dim": int(embedding_dim),
    }


def _prefix_texts(texts: List[str], input_type: str) -> List[str]:
    normalized_type = input_type.lower()
    if normalized_type == "query":
        prefix = "query:"
    elif normalized_type in {"document", "doc", "passage"}:
        prefix = "passage:"
    else:
        prefix = f"{normalized_type}:"
    return [f"{prefix} {text}" for text in texts]


def _tokenize(
    texts: List[str],
    assets: Dict[str, Any],
    truncate: TruncationMode,
) -> Dict[str, torch.Tensor]:
    tokenizer: AutoTokenizer = assets["tokenizer"]
    truncation_side = "left" if truncate == "START" else "right"
    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    try:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=truncate != "NONE",
            max_length=assets["max_length"],
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = previous_truncation_side
    return {key: value.to(assets["device"]) for key, value in encoded.items()}


def _average_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    masked = last_hidden_state.masked_fill(~mask.bool(), 0.0)
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return F.normalize(summed / counts, p=2, dim=-1)


def _maybe_downproject(
    embeddings: torch.Tensor,
    requested_dim: Optional[int],
    full_dim: int,
) -> torch.Tensor:
    if requested_dim is None:
        return embeddings
    if requested_dim > full_dim:
        raise HTTPException(
            status_code=400,
            detail=f"dimensions {requested_dim} exceeds embedding size {full_dim}",
        )
    sliced = embeddings[:, :requested_dim]
    return F.normalize(sliced, p=2, dim=-1)


def _to_base64(vector: torch.Tensor) -> str:
    float_bytes = vector.cpu().numpy().astype("float32").tobytes()
    return base64.b64encode(float_bytes).decode("ascii")


def _validate_model_name(model_name: Optional[str]) -> None:
    if model_name is None:
        return
    if model_name != settings.model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}', expected '{settings.model_id}'",
        )


def _validate_batch_size(item_count: int, limit: int) -> None:
    if item_count > limit:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {item_count} exceeds limit {limit}",
        )


def _run_model(encoded: Dict[str, torch.Tensor], dims: Optional[int], assets: Dict[str, Any]) -> torch.Tensor:
    model: AutoModel = assets["model"]
    try:
        with torch.inference_mode():
            outputs = model(**encoded)

        embeddings = _average_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = embeddings.to(torch.float32)
        embeddings = _maybe_downproject(
            embeddings,
            dims,
            assets["embedding_dim"],
        )
        return embeddings
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to caller
        logger.exception("Embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail="Embedding failed") from exc


def _embed(request_body: EmbeddingsRequest, batch_limit: int) -> Dict[str, Any]:
    _validate_model_name(request_body.model)
    _validate_batch_size(len(request_body.input), batch_limit)
    start_time = time.perf_counter()
    with _model_usage_http() as assets:
        prefixed_inputs = _prefix_texts(request_body.input, request_body.input_type)
        try:
            encoded = _tokenize(prefixed_inputs, assets, request_body.truncate)
        except ValueError as exc:
            logger.error("Tokenization failed: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        embeddings = _run_model(encoded, request_body.dimensions, assets)

        data: List[Dict[str, Any]] = []
        for index, vector in enumerate(embeddings):
            embedding_value = vector.cpu().numpy().astype("float32")
            if request_body.encoding_format == "base64":
                embedding_payload = _to_base64(torch.from_numpy(embedding_value))
            else:
                embedding_payload = embedding_value.tolist()
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding_payload,
                    "model": settings.model_id,
                }
            )

        token_count = int(encoded["attention_mask"].sum().item())
        response = {
            "object": "list",
            "model": settings.model_id,
            "data": data,
            "usage": {"prompt_tokens": token_count, "total_tokens": token_count},
        }
    _record_inference_success(int((time.perf_counter() - start_time) * 1e9))
    return response


def _record_metrics(endpoint: str, protocol: str, status: str, started: float) -> None:
    duration = time.perf_counter() - started
    REQUEST_COUNTER.labels(protocol=protocol, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol=protocol, endpoint=endpoint).observe(duration)


def _deserialize_bytes_input(
    tensor: service_pb2.ModelInferRequest.InferInputTensor, raw_contents: Optional[bytes]
) -> List[str]:
    if raw_contents:
        values = deserialize_bytes_tensor(raw_contents)
    elif tensor.contents.raw_contents:
        values = deserialize_bytes_tensor(tensor.contents.raw_contents)
    elif tensor.contents.bytes_contents:
        values = np.array(list(tensor.contents.bytes_contents), dtype=np.object_)
    else:
        values = np.array([], dtype=np.object_)
    flattened = values.reshape(-1)
    return [
        item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        for item in flattened.tolist()
    ]


def _parse_dimensions_from_tensor(
    tensor: service_pb2.ModelInferRequest.InferInputTensor, raw_contents: Optional[bytes]
) -> Optional[int]:
    if tensor.datatype not in {"INT32", "INT64"}:
        return None
    if raw_contents:
        array = np.frombuffer(raw_contents, dtype=np.int64 if tensor.datatype == "INT64" else np.int32)
    elif tensor.contents.raw_contents:
        array = np.frombuffer(tensor.contents.raw_contents, dtype=np.int64 if tensor.datatype == "INT64" else np.int32)
    elif tensor.contents.int64_contents:
        array = np.array(list(tensor.contents.int64_contents), dtype=np.int64)
    elif tensor.contents.int_contents:
        array = np.array(list(tensor.contents.int_contents), dtype=np.int32)
    else:
        return None
    if array.size == 0:
        return None
    value = int(array.reshape(-1)[0])
    return value if value > 0 else None


def _parse_shared_memory_params(
    parameters: Mapping[str, service_pb2.InferParameter],
) -> Tuple[Optional[str], Optional[int], int]:
    region_param = parameters.get("shared_memory_region")
    region_name = None
    if region_param is not None and region_param.WhichOneof("parameter_choice") == "string_param":
        region_name = region_param.string_param or None

    byte_size = None
    byte_size_param = parameters.get("shared_memory_byte_size")
    if byte_size_param is not None and byte_size_param.WhichOneof("parameter_choice"):
        byte_size = int(byte_size_param.int64_param or byte_size_param.uint64_param)

    offset = 0
    offset_param = parameters.get("shared_memory_offset")
    if offset_param is not None and offset_param.WhichOneof("parameter_choice"):
        offset = int(offset_param.int64_param or offset_param.uint64_param)
        if offset < 0:
            raise ValueError("shared_memory_offset must be non-negative")

    return region_name, byte_size, offset


def _get_cudart() -> ctypes.CDLL:
    global _cudart
    if _cudart is not None:
        return _cudart
    library_path = ctypes.util.find_library("cudart")
    candidates = [library_path] if library_path else []
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.dylib",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            _cudart = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if _cudart is None:
        raise RuntimeError("CUDA runtime library not found")
    _cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
    _cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
    _cudart.cudaMemcpy.restype = ctypes.c_int
    _cudart.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _cudart.cudaSetDevice.restype = ctypes.c_int
    _cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    return _cudart


def _check_cuda(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error code {result}")


def _close_cuda_pointer(pointer: int | None, device_id: int) -> None:
    if pointer is None:
        return
    try:
        cudart = _get_cudart()
        _check_cuda(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")
        _check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)), "cudaIpcCloseMemHandle")
    except Exception:  # pragma: no cover - best effort cleanup
        logger.debug("Failed to close CUDA shared memory handle", exc_info=True)


def _read_cuda_shared_memory(region_name: str, byte_size: int, offset: int) -> bytes:
    with _shared_memory_lock:
        if region_name not in _cuda_shared_memory:
            raise KeyError(region_name)
        region = _cuda_shared_memory[region_name]
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + byte_size
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    cudart = _get_cudart()
    _check_cuda(cudart.cudaSetDevice(int(region.device_id)), "cudaSetDevice")
    host_buffer = (ctypes.c_char * byte_size)()
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda(
        cudart.cudaMemcpy(
            ctypes.cast(host_buffer, ctypes.c_void_p),
            device_ptr,
            ctypes.c_size_t(byte_size),
            CUDA_MEMCPY_DEVICE_TO_HOST,
        ),
        "cudaMemcpy (device to host)",
    )
    return bytes(host_buffer)


def _write_cuda_shared_memory(region_name: str, data: bytes, offset: int, byte_size: int) -> None:
    with _shared_memory_lock:
        if region_name not in _cuda_shared_memory:
            raise KeyError(region_name)
        region = _cuda_shared_memory[region_name]
    if not data:
        return
    if byte_size <= 0:
        raise ValueError("shared_memory_byte_size must be positive")
    if len(data) > byte_size:
        raise ValueError("Shared memory region too small for data")
    if region.pointer is None:
        raise RuntimeError("CUDA shared memory pointer is not available")
    start = region.offset + offset
    end = start + len(data)
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    cudart = _get_cudart()
    _check_cuda(cudart.cudaSetDevice(int(region.device_id)), "cudaSetDevice")
    host_buffer = ctypes.create_string_buffer(data)
    device_ptr = ctypes.c_void_p(region.pointer + start)
    _check_cuda(
        cudart.cudaMemcpy(
            device_ptr,
            ctypes.cast(host_buffer, ctypes.c_void_p),
            ctypes.c_size_t(len(data)),
            CUDA_MEMCPY_HOST_TO_DEVICE,
        ),
        "cudaMemcpy (host to device)",
    )


def _read_system_shared_memory(region_name: str, byte_size: int, offset: int) -> bytes:
    with _shared_memory_lock:
        if region_name not in _system_shared_memory:
            raise KeyError(region_name)
        region = _system_shared_memory[region_name]
    start = region.offset + offset
    end = start + byte_size
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory read out of bounds")
    return bytes(region.buffer[start:end])


def _write_system_shared_memory(region_name: str, data: bytes, offset: int, byte_size: int) -> None:
    with _shared_memory_lock:
        if region_name not in _system_shared_memory:
            raise KeyError(region_name)
        region = _system_shared_memory[region_name]
    start = region.offset + offset
    end = start + byte_size
    if start < 0 or end > region.offset + region.byte_size:
        raise ValueError("Shared memory write out of bounds")
    if len(data) > byte_size:
        raise ValueError("Shared memory region too small for data")
    region.buffer[start:start + len(data)] = data


def _register_system_shared_memory(name: str, key: str, offset: int, byte_size: int) -> None:
    if not name:
        raise ValueError("name is required")
    if not key:
        raise ValueError("key is required")
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    shm_obj: shared_memory.SharedMemory | None = None
    mmap_obj: mmap.mmap | None = None
    file_handle: object | None = None
    buffer_size: int
    with _shared_memory_lock:
        if name in _system_shared_memory or name in _cuda_shared_memory:
            raise FileExistsError(f"Region {name} already registered")
    try:
        try:
            shm_obj = shared_memory.SharedMemory(name=key, create=False)
            buffer_size = shm_obj.size
        except FileNotFoundError:
            path = key[7:] if key.startswith("file://") else key
            file_handle = open(path, "r+b")
            file_handle.seek(0, os.SEEK_END)
            buffer_size = file_handle.tell()
            file_handle.seek(0)
            mmap_obj = mmap.mmap(file_handle.fileno(), length=0, access=mmap.ACCESS_WRITE)
        if offset < 0 or offset + byte_size > buffer_size:
            raise ValueError("Shared memory region too small for requested window")
        region = SystemSharedMemoryRegion(
            name=name,
            key=key,
            offset=offset,
            byte_size=byte_size,
            shm=shm_obj,
            mmap_obj=mmap_obj,
            file_handle=file_handle,
        )
        with _shared_memory_lock:
            _system_shared_memory[name] = region
    except Exception:
        if shm_obj is not None:
            shm_obj.close()
        if mmap_obj is not None:
            mmap_obj.close()
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception:
                logger.debug("Failed to close shared memory file handle", exc_info=True)
        raise


def _unregister_system_shared_memory(name: str) -> None:
    with _shared_memory_lock:
        region = _system_shared_memory.pop(name, None)
    if region:
        region.close()


def _register_cuda_shared_memory(name: str, raw_handle: bytes, device_id: int, byte_size: int, offset: int) -> None:
    if not name:
        raise ValueError("name is required")
    if not raw_handle:
        raise ValueError("raw_handle is required for CUDA shared memory registration")
    if byte_size <= 0:
        raise ValueError("byte_size must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    with _shared_memory_lock:
        if name in _cuda_shared_memory or name in _system_shared_memory:
            raise FileExistsError(f"Region {name} already registered")
    pointer: int | None = None
    cudart: ctypes.CDLL | None = None
    try:
        cudart = _get_cudart()
        _check_cuda(cudart.cudaSetDevice(int(device_id)), "cudaSetDevice")
        handle_buffer = ctypes.create_string_buffer(raw_handle)
        device_pointer = ctypes.c_void_p()
        _check_cuda(
            cudart.cudaIpcOpenMemHandle(
                ctypes.byref(device_pointer),
                ctypes.cast(handle_buffer, ctypes.c_void_p),
                CUDA_IPC_FLAG,
            ),
            "cudaIpcOpenMemHandle",
        )
        if device_pointer.value is None:
            raise RuntimeError("cudaIpcOpenMemHandle returned null pointer")
        pointer = int(device_pointer.value)
        region = CudaSharedMemoryRegion(
            name=name,
            device_id=int(device_id),
            byte_size=int(byte_size),
            raw_handle=bytes(raw_handle),
            offset=int(offset),
            pointer=pointer,
        )
        with _shared_memory_lock:
            if name in _cuda_shared_memory or name in _system_shared_memory:
                raise FileExistsError(f"Region {name} already registered")
            _cuda_shared_memory[name] = region
    except Exception:
        if pointer is not None and cudart is not None:
            try:
                _check_cuda(cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(pointer)), "cudaIpcCloseMemHandle")
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("Failed to close CUDA shared memory handle", exc_info=True)
        raise


def _unregister_cuda_shared_memory(name: str) -> None:
    with _shared_memory_lock:
        region = _cuda_shared_memory.pop(name, None)
    if region is not None:
        _close_cuda_pointer(region.pointer, region.device_id)


app = FastAPI(title=settings.model_id, version=settings.model_version)
app.state.assets = None
_load_model()
MAX_BATCH_SIZE = _read_max_batch_size()
triton_config = TritonModelConfig(
    name=settings.model_id,
    version=settings.model_version,
    max_batch_size=MAX_BATCH_SIZE,
    embedding_dim=app.state.assets["embedding_dim"],
)


@app.middleware("http")
async def _http_tracing(request: Request, call_next):
    endpoint = request.url.path
    status = "200"
    if TRACER:
        span_name = f"http {request.method.lower()} {endpoint}"
        ctx_manager = TRACER.start_as_current_span(span_name)
    else:
        ctx_manager = nullcontext()
    start_time = time.perf_counter()
    INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).inc()
    try:
        with ctx_manager:
            response = await call_next(request)
            status = str(response.status_code)
            return response
    except HTTPException as exc:
        status = str(exc.status_code)
        raise
    except Exception:
        status = "500"
        raise
    finally:
        INFLIGHT_GAUGE.labels(protocol="http", endpoint=endpoint).dec()
        _record_metrics(endpoint, "http", status, start_time)


@app.get("/v1/health/live", dependencies=[Depends(_require_http_auth)])
async def live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v1/health/ready", dependencies=[Depends(_require_http_auth)])
async def ready() -> Dict[str, bool]:
    return {"ready": _is_model_ready()}


@app.get("/v1/models", dependencies=[Depends(_require_http_auth)])
async def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": settings.model_id}]}


@app.get("/v1/metadata", dependencies=[Depends(_require_http_auth)])
async def metadata() -> Dict[str, Any]:
    short_name = f"{settings.model_id}:{settings.model_version}"
    return {
        "id": settings.model_id,
        "name": settings.model_id,
        "version": settings.model_version,
        "modelInfo": [
            {
                "name": settings.model_id,
                "version": settings.model_version,
                "shortName": short_name,
            }
        ],
    }


@app.post("/v1/embeddings", dependencies=[Depends(_require_http_auth)])
async def embeddings(request_body: EmbeddingsRequest) -> Dict[str, Any]:
    async with async_rate_limiter.limit():
        return _embed(request_body, triton_config.max_batch_size)


@app.get("/metrics", dependencies=[Depends(_require_http_auth)])
async def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", dependencies=[Depends(_require_http_auth)])
async def root() -> JSONResponse:
    return JSONResponse({"message": settings.model_id}, status_code=200)


def _proto_to_dict(message: Any) -> Dict[str, Any]:
    return json_format.MessageToDict(message, preserving_proto_field_name=True)


def _validate_model_name_param(model_name: str) -> None:
    if model_name != triton_config.name:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")


def _validate_grpc_model_name(model_name: str, context: grpc.ServicerContext) -> None:
    if model_name and model_name != triton_config.name:
        context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model '{model_name}'")


@app.get("/v2/health/live", dependencies=[Depends(_require_http_auth)])
async def triton_live() -> Dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready", dependencies=[Depends(_require_http_auth)])
async def triton_ready() -> Dict[str, bool]:
    return {"ready": _is_model_ready()}


@app.get("/v2/models", dependencies=[Depends(_require_http_auth)])
async def list_triton_models() -> Dict[str, Any]:
    index = _model_repository_index()
    return _proto_to_dict(index)


@app.get("/v2/models/{model_name}", dependencies=[Depends(_require_http_auth)])
async def model_metadata_http(model_name: str) -> Dict[str, Any]:
    _validate_model_name_param(model_name)
    proto = _model_metadata_response()
    return _proto_to_dict(proto)


@app.get("/v2/models/{model_name}/config", dependencies=[Depends(_require_http_auth)])
async def model_config_http(model_name: str) -> Dict[str, Any]:
    _validate_model_name_param(model_name)
    proto = _model_config_response()
    return _proto_to_dict(proto)


@app.get("/v2/models/{model_name}/ready", dependencies=[Depends(_require_http_auth)])
async def model_ready_http(model_name: str) -> Dict[str, bool]:
    _validate_model_name_param(model_name)
    return {"ready": _is_model_ready()}


def _server_live_response() -> service_pb2.ServerLiveResponse:
    return service_pb2.ServerLiveResponse(live=True)


def _server_ready_response() -> service_pb2.ServerReadyResponse:
    return service_pb2.ServerReadyResponse(ready=_is_model_ready())


def _model_ready_response() -> service_pb2.ModelReadyResponse:
    return service_pb2.ModelReadyResponse(ready=_is_model_ready())


def _model_metadata_response() -> service_pb2.ModelMetadataResponse:
    tensor_inputs = [
        service_pb2.ModelMetadataResponse.TensorMetadata(name=TEXT_INPUT_NAME, datatype="BYTES", shape=[-1]),
        service_pb2.ModelMetadataResponse.TensorMetadata(name=DIMENSIONS_INPUT_NAME, datatype="INT64", shape=[1]),
    ]
    tensor_outputs = [
        service_pb2.ModelMetadataResponse.TensorMetadata(
            name=OUTPUT_NAME, datatype="FP32", shape=[-1, triton_config.embedding_dim]
        )
    ]
    return service_pb2.ModelMetadataResponse(
        name=triton_config.name,
        versions=[triton_config.version],
        platform="python",
        inputs=tensor_inputs,
        outputs=tensor_outputs,
    )


def _model_config_response() -> model_config_pb2.ModelConfigResponse:
    config = model_config_pb2.ModelConfig(
        name=triton_config.name,
        platform="python",
        max_batch_size=triton_config.max_batch_size,
        input=[
            model_config_pb2.ModelInput(
                name=TEXT_INPUT_NAME,
                data_type=model_config_pb2.TYPE_BYTES,
                dims=[-1],
            ),
            model_config_pb2.ModelInput(
                name=DIMENSIONS_INPUT_NAME,
                data_type=model_config_pb2.TYPE_INT64,
                dims=[1],
            ),
        ],
        output=[
            model_config_pb2.ModelOutput(
                name=OUTPUT_NAME,
                data_type=model_config_pb2.TYPE_FP32,
                dims=[-1, triton_config.embedding_dim],
            )
        ],
    )
    return model_config_pb2.ModelConfigResponse(config=config)


def _model_repository_index() -> service_pb2.RepositoryIndexResponse:
    index = service_pb2.RepositoryIndexResponse()
    model = index.models.add()
    model.name = triton_config.name
    model.version = triton_config.version
    ready = _is_model_ready()
    model.state = "READY" if ready else "UNAVAILABLE"
    model.reason = "" if ready else "Model is unloaded"
    return index


def _run_embedding_array(
    texts: List[str],
    dims: Optional[int],
    input_type: str = "passage",
    truncate: TruncationMode = "END",
    assets: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    request = EmbeddingsRequest(
        model=settings.model_id,
        input=texts,
        encoding_format="float",
        input_type=input_type,
        truncate=truncate,
        dimensions=dims,
    )
    _validate_batch_size(len(request.input), triton_config.max_batch_size)
    resolved_assets = assets or app.state.assets
    if resolved_assets is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    prefixed_inputs = _prefix_texts(request.input, request.input_type)
    encoded = _tokenize(prefixed_inputs, resolved_assets, request.truncate)
    embeddings = _run_model(encoded, request.dimensions, resolved_assets)
    return embeddings.cpu().numpy().astype("float32")


class TritonService(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, state: ModelState):
        super().__init__()
        self._state = state

    def ServerLive(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        return _server_live_response()

    def ServerReady(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        return _server_ready_response()

    def ModelReady(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        _validate_grpc_model_name(request.name, context)
        return _model_ready_response()

    def ServerMetadata(self, request, context):  # noqa: N802
        return service_pb2.ServerMetadataResponse(
            name=settings.model_id,
            version=settings.model_version,
            extensions=["model_repository", "metrics"],
        )

    def ModelMetadata(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        _validate_grpc_model_name(request.name, context)
        return _model_metadata_response()

    def ModelConfig(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        _validate_grpc_model_name(request.name, context)
        return _model_config_response()

    def RepositoryIndex(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        return _model_repository_index()

    def RepositoryModelLoad(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        _validate_grpc_model_name(request.model_name, context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            _load_model()
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("Model load failed: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to load model")
        return service_pb2.RepositoryModelLoadResponse()

    def RepositoryModelUnload(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        _validate_grpc_model_name(request.model_name, context)
        if not settings.enable_model_control:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Model control disabled")
        try:
            _unload_model()
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("Model unload failed: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to unload model")
        return service_pb2.RepositoryModelUnloadResponse()

    def _infer_once(self, request, context, endpoint: str) -> service_pb2.ModelInferResponse:
        start_time = time.perf_counter()
        _require_grpc_auth(context)
        if not _is_model_ready():
            _record_metrics(endpoint, "grpc", "not_loaded", start_time)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Model is not loaded")
        if request.model_name:
            _validate_grpc_model_name(request.model_name, context)

        input_lookup: Dict[str, int] = {tensor.name: idx for idx, tensor in enumerate(request.inputs)}
        if TEXT_INPUT_NAME not in input_lookup:
            _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{TEXT_INPUT_NAME} missing")

        def _raw_for_tensor(tensor: service_pb2.ModelInferRequest.InferInputTensor, index: int) -> Optional[bytes]:
            if index < len(request.raw_input_contents):
                return request.raw_input_contents[index]
            try:
                region_name, region_size, region_offset = _parse_shared_memory_params(tensor.parameters)
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            if region_name is None:
                return None
            if region_size is None:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required")
            if region_size <= 0:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size must be positive")
            try:
                with _shared_memory_lock:
                    is_cuda = region_name in _cuda_shared_memory
                    is_system = region_name in _system_shared_memory
                if not is_cuda and not is_system:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Shared memory region '{region_name}' not found")
                if is_cuda:
                    return _read_cuda_shared_memory(region_name, int(region_size), int(region_offset))
                return _read_system_shared_memory(region_name, int(region_size), int(region_offset))
            except KeyError:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.NOT_FOUND, f"Shared memory region '{region_name}' not found")
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                _record_metrics(endpoint, "grpc", "error", start_time)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

        text_index = input_lookup[TEXT_INPUT_NAME]
        text_tensor = request.inputs[text_index]
        text_raw = _raw_for_tensor(text_tensor, text_index)
        texts = [s for s in _deserialize_bytes_input(text_tensor, text_raw) if s]
        if not texts:
            _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No input provided")

        dims_value: Optional[int] = None
        if DIMENSIONS_INPUT_NAME in input_lookup:
            dim_idx = input_lookup[DIMENSIONS_INPUT_NAME]
            dim_tensor = request.inputs[dim_idx]
            dim_raw = _raw_for_tensor(dim_tensor, dim_idx)
            dims_value = _parse_dimensions_from_tensor(dim_tensor, dim_raw)

        if len(texts) > triton_config.max_batch_size:
            _record_metrics(endpoint, "grpc", "batch_limit", start_time)
            context.abort(
                grpc.StatusCode.OUT_OF_RANGE,
                f"Batch size {len(texts)} exceeds limit {triton_config.max_batch_size}",
            )

        output_region: Optional[str] = None
        output_offset = 0
        output_byte_size: Optional[int] = None
        output_is_cuda = False
        if request.outputs:
            output_tensor = request.outputs[0]
            try:
                region_name, region_size, region_offset = _parse_shared_memory_params(output_tensor.parameters)
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            if region_name:
                with _shared_memory_lock:
                    output_is_cuda = region_name in _cuda_shared_memory
                    output_is_system = region_name in _system_shared_memory
                if not output_is_cuda and not output_is_system:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.NOT_FOUND, f"Shared memory region '{region_name}' not found")
                if region_size is None:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size is required for outputs")
                if region_size <= 0:
                    _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "shared_memory_byte_size must be positive")
                output_region = region_name
                output_offset = int(region_offset)
                output_byte_size = int(region_size)

        INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).inc()
        try:
            tracer_ctx = TRACER.start_as_current_span("grpc_infer") if TRACER else nullcontext()
            with tracer_ctx:
                with sync_rate_limiter.limit():
                    with _model_usage_grpc(context) as assets:
                        vectors = _run_embedding_array(texts, dims_value, assets=assets)
        except HTTPException as exc:
            _record_metrics(endpoint, "grpc", str(exc.status_code), start_time)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, exc.detail)
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("gRPC embedding failed: %s", exc)
            _record_metrics(endpoint, "grpc", "error", start_time)
            context.abort(grpc.StatusCode.INTERNAL, "Embedding failed")
        finally:
            INFLIGHT_GAUGE.labels(protocol="grpc", endpoint=endpoint).dec()

        serialized = vectors.tobytes()
        output = service_pb2.ModelInferResponse.InferOutputTensor(
            name=OUTPUT_NAME,
            datatype="FP32",
            shape=[len(texts), vectors.shape[1]],
        )
        response = service_pb2.ModelInferResponse(
            model_name=settings.model_id,
            outputs=[output],
            id=request.id,
        )
        if output_region is not None and output_byte_size is not None:
            if len(serialized) > output_byte_size:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Shared memory region '{output_region}' is too small for output",
                )
            try:
                if output_is_cuda:
                    _write_cuda_shared_memory(output_region, serialized, output_offset, output_byte_size)
                else:
                    _write_system_shared_memory(output_region, serialized, output_offset, output_byte_size)
                response.raw_output_contents[:] = []
            except KeyError:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.NOT_FOUND, f"Shared memory region '{output_region}' not found")
            except ValueError as exc:
                _record_metrics(endpoint, "grpc", "invalid_payload", start_time)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except RuntimeError as exc:
                _record_metrics(endpoint, "grpc", "error", start_time)
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
        else:
            response.raw_output_contents.append(serialized)
        duration_ns = int((time.perf_counter() - start_time) * 1e9)
        _record_metrics(endpoint, "grpc", "success", start_time)
        _record_inference_success(duration_ns)
        return response

    def ModelInfer(self, request, context):  # noqa: N802
        return self._infer_once(request, context, "grpc::infer")

    def ModelStatistics(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        if request.name and request.name != triton_config.name:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Unknown model '{request.name}'")
        with _model_condition:
            inference_count = model_state.inference_count
            last_inference = model_state.last_inference_ns
            success_ns_total = model_state.success_ns_total
        stats = service_pb2.ModelStatistics(
            name=triton_config.name,
            version=triton_config.version,
            last_inference=last_inference,
            inference_count=inference_count,
            execution_count=inference_count,
        )
        stats.inference_stats.success.count = inference_count
        stats.inference_stats.success.ns = success_ns_total
        return service_pb2.ModelStatisticsResponse(model_stats=[stats])

    def ModelStreamInfer(self, request_iterator, context):  # noqa: N802
        for infer_request in request_iterator:
            yield self._infer_once(infer_request, context, "grpc::stream_infer")

    def SystemSharedMemoryStatus(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        response = service_pb2.SystemSharedMemoryStatusResponse()
        with _shared_memory_lock:
            if request.name:
                if request.name not in _system_shared_memory:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"System shared memory region '{request.name}' not found")
                regions = [request.name]
            else:
                regions = list(_system_shared_memory.keys())
            for region_name in regions:
                region = _system_shared_memory[region_name]
                status = response.regions[region.name]
                status.name = region.name
                status.key = region.key
                status.offset = region.offset
                status.byte_size = region.byte_size
        return response

    def SystemSharedMemoryRegister(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        try:
            _register_system_shared_memory(request.name, request.key, int(request.offset), int(request.byte_size))
            return service_pb2.SystemSharedMemoryRegisterResponse()
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except FileNotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to client
            logger.exception("Failed to register system shared memory: %s", exc)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to register system shared memory")

    def SystemSharedMemoryUnregister(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        with _shared_memory_lock:
            if request.name not in _system_shared_memory:
                context.abort(grpc.StatusCode.NOT_FOUND, f"System shared memory region '{request.name}' not found")
        _unregister_system_shared_memory(request.name)
        return service_pb2.SystemSharedMemoryUnregisterResponse()

    def CudaSharedMemoryStatus(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        response = service_pb2.CudaSharedMemoryStatusResponse()
        with _shared_memory_lock:
            if request.name:
                if request.name not in _cuda_shared_memory:
                    context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' not found")
                regions = [request.name]
            else:
                regions = list(_cuda_shared_memory.keys())
            for region_name in regions:
                region = _cuda_shared_memory[region_name]
                status = response.regions[region.name]
                status.name = region.name
                status.device_id = region.device_id
                status.byte_size = region.byte_size
        return response

    def CudaSharedMemoryRegister(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        try:
            _register_cuda_shared_memory(
                request.name,
                bytes(request.raw_handle),
                int(request.device_id),
                int(request.byte_size),
                int(request.offset),
            )
            return service_pb2.CudaSharedMemoryRegisterResponse()
        except FileExistsError as exc:
            context.abort(grpc.StatusCode.ALREADY_EXISTS, str(exc))
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except RuntimeError as exc:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))

    def CudaSharedMemoryUnregister(self, request, context):  # noqa: N802
        _require_grpc_auth(context)
        if not request.name:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "name is required")
        with _shared_memory_lock:
            if request.name not in _cuda_shared_memory:
                context.abort(grpc.StatusCode.NOT_FOUND, f"CUDA shared memory region '{request.name}' not found")
        _unregister_cuda_shared_memory(request.name)
        return service_pb2.CudaSharedMemoryUnregisterResponse()


grpc_server: grpc.Server | None = None


def _start_grpc_server() -> grpc.Server:
    workers = max(settings.rate_limit or 4, 4)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(TritonService(model_state), server)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    server.start()
    logger.info("gRPC server listening on %s", settings.grpc_port)
    return server


@app.on_event("startup")
def _startup() -> None:
    global grpc_server
    if grpc_server is None:
        if futures is None:
            raise RuntimeError("concurrent.futures is required for gRPC server")
        grpc_server = _start_grpc_server()


@app.on_event("shutdown")
def _shutdown() -> None:
    global grpc_server
    if grpc_server:
        grpc_server.stop(grace=0)
        grpc_server = None
    _unload_model()
