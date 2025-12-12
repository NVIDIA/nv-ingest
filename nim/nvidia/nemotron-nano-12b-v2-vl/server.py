from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, start_http_server
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextIteratorStreamer

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


logger = logging.getLogger("nemotron-nano-12b-v2-vl")


DEFAULT_SYSTEM_PROMPT = "/no_think"
DEFAULT_USER_PROMPT = "Caption the content of this image:"
MAX_DATA_URL_CHARS = 200_000


class ServiceSettings(BaseSettings):
    model_id: str = Field("nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", alias="MODEL_ID")
    served_model_name: str = Field("nvidia/nemotron-nano-12b-v2-vl", alias="SERVED_MODEL_NAME")
    model_version: str = Field("1.0.0", alias="MODEL_VERSION")
    http_port: int = Field(8000, alias="NIM_HTTP_API_PORT")
    metrics_port: int | None = Field(8002, alias="NIM_METRICS_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_verbose: int = Field(0, alias="NIM_TRITON_LOG_VERBOSE")
    rate_limit: int | None = Field(None, alias="NIM_TRITON_RATE_LIMIT")
    require_auth: bool = Field(True, alias="NIM_REQUIRE_AUTH")
    auth_token: str | None = Field(None, alias="NGC_API_KEY")
    fallback_auth_token: str | None = Field(None, alias="NIM_NGC_API_KEY")
    nvidia_auth_token: str | None = Field(None, alias="NVIDIA_API_KEY")
    enable_otel: bool = Field(False, alias="NIM_ENABLE_OTEL")
    otel_endpoint: str | None = Field(None, alias="NIM_OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_service_name: str | None = Field(None, alias="NIM_OTEL_SERVICE_NAME")
    default_max_tokens: int = Field(512, alias="DEFAULT_MAX_TOKENS")
    max_output_tokens: int = Field(1024, alias="MAX_OUTPUT_TOKENS")
    max_batch_size: int = Field(1, alias="NIM_TRITON_MAX_BATCH_SIZE")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="VLM_CAPTION_SYSTEM_PROMPT")
    user_prompt: str = Field(DEFAULT_USER_PROMPT, alias="VLM_CAPTION_PROMPT")
    video_pruning_rate: float = Field(0.0, alias="VIDEO_PRUNING_RATE")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    @property
    def resolved_auth_tokens(self) -> set[str]:
        return {token for token in (self.auth_token, self.fallback_auth_token, self.nvidia_auth_token) if token}


settings = ServiceSettings()
logging.basicConfig(level=logging.DEBUG if settings.log_verbose else logging.INFO)
logger.setLevel(settings.log_level.upper())


def _configure_tracing() -> Optional["trace.Tracer"]:
    if not settings.enable_otel or not trace or not otel_metrics:
        return None
    try:
        resource = Resource.create({"service.name": settings.otel_service_name or settings.served_model_name})
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_endpoint or "http://localhost:4318/v1/traces", insecure=True)
            )
        )
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=settings.otel_endpoint or "http://localhost:4318/v1/metrics", insecure=True)
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        otel_metrics.set_meter_provider(meter_provider)
        return trace.get_tracer(settings.otel_service_name or settings.served_model_name)
    except Exception:  # pragma: no cover - surfaced to logs
        logger.exception("Failed to configure OTEL")
        return None


TRACER = _configure_tracing()


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    return torch.float32


@dataclass
class ModelAssets:
    model: Any
    tokenizer: Any
    processor: Any
    device: str
    dtype: torch.dtype


@dataclass
class ModelState:
    loaded: bool = False
    last_error: str | None = None


def _load_model() -> ModelAssets:
    device = _select_device()
    dtype = _select_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained(settings.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    if hasattr(model, "video_pruning_rate"):
        try:
            model.video_pruning_rate = max(settings.video_pruning_rate, 0.0)
        except Exception:
            logger.debug("Model does not expose video_pruning_rate setter; skipping")

    logger.info("Loaded %s to %s (dtype=%s)", settings.model_id, device, dtype)
    return ModelAssets(model=model, tokenizer=tokenizer, processor=processor, device=device, dtype=dtype)


model_state = ModelState()
assets: ModelAssets | None
try:
    assets = _load_model()
    model_state.loaded = True
except Exception as exc:  # pragma: no cover - surfaced at startup
    logger.exception("Failed to load model")
    model_state.loaded = False
    model_state.last_error = str(exc)
    assets = None


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


async_rate_limiter = AsyncRateLimiter(settings.rate_limit)


REQUEST_COUNTER = Counter("nim_requests_total", "Total requests served", ["protocol", "endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "nim_request_latency_seconds",
    "Latency for requests",
    ["protocol", "endpoint"],
    buckets=(
        0.01,
        0.02,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ),
)
INFLIGHT_GAUGE = Gauge("nim_requests_in_flight", "Concurrent in-flight requests", ["protocol", "endpoint"])

_metrics_started = False
if settings.metrics_port and not _metrics_started:
    start_http_server(settings.metrics_port)
    _metrics_started = True
    logger.info("Metrics server listening on port %s", settings.metrics_port)


class ImageURL(BaseModel):
    url: str
    detail: str | None = None

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("image_url.url must not be empty")
        return stripped


class MessageContent(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = Field(None, alias="image_url")

    @model_validator(mode="after")
    def _validate_payload(self) -> "MessageContent":
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text content must not be empty")
        elif self.type == "image_url":
            if self.image_url is None:
                raise ValueError("image_url content is required for image_url type")
        return self


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | List[MessageContent]

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str | List[MessageContent]) -> str | List[MessageContent]:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("content must not be empty")
            return value
        if isinstance(value, Sequence) and value:
            return list(value)
        raise ValueError("content must be a string or non-empty list")


class ChatRequest(BaseModel):
    model: str | None = None
    messages: List[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def _validate_messages(cls, value: List[ChatMessage]) -> List[ChatMessage]:
        if not value:
            raise ValueError("messages must contain at least one entry")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("temperature must be non-negative")
        return value

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, value: float) -> float:
        if value <= 0.0 or value > 1.0:
            raise ValueError("top_p must be in (0, 1]")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_tokens must be positive")
        return value


class ChatChoice(BaseModel):
    index: int
    message: Mapping[str, str]
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]


def _extract_token(header_value: str | None) -> str | None:
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


def _is_authorized(header_value: str | None) -> bool:
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


def _decode_image_url(data_url: str) -> Image.Image:
    if len(data_url) > MAX_DATA_URL_CHARS:
        raise HTTPException(status_code=400, detail="image payload too large")
    if not data_url.startswith("data:"):
        raise HTTPException(status_code=400, detail="only data URL images are supported")
    try:
        _, encoded = data_url.split(",", maxsplit=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid data URL") from exc
    try:
        binary = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image content") from exc
    try:
        image = Image.open(io.BytesIO(binary))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail="unable to decode image") from exc


@dataclass
class ParsedRequest:
    prompt_text: str
    image: Image.Image
    max_new_tokens: int
    temperature: float
    top_p: float
    stream: bool


def _coerce_prompts(messages: List[ChatMessage]) -> tuple[str, str, Image.Image]:
    system_prompt = settings.system_prompt or DEFAULT_SYSTEM_PROMPT
    user_prompt_parts: List[str] = []
    decoded_image: Image.Image | None = None
    image_count = 0

    for message in messages:
        if message.role == "system" and isinstance(message.content, str):
            system_prompt = message.content.strip() or system_prompt
            continue
        if message.role != "user":
            continue
        contents = message.content
        if isinstance(contents, str):
            user_prompt_parts.append(contents.strip())
            continue
        for item in contents:
            if item.type == "text" and item.text:
                user_prompt_parts.append(item.text.strip())
            elif item.type == "image_url" and item.image_url and decoded_image is None:
                decoded_image = _decode_image_url(item.image_url.url)
                image_count += 1
            elif item.type == "image_url" and item.image_url:
                image_count += 1

    if decoded_image is None:
        raise HTTPException(status_code=400, detail="exactly one image is required")
    if image_count > settings.max_batch_size:
        raise HTTPException(status_code=400, detail="too many images provided")
    if len(user_prompt_parts) == 0:
        user_prompt_parts.append(settings.user_prompt or DEFAULT_USER_PROMPT)

    user_prompt = "\n".join(part for part in user_prompt_parts if part)
    return system_prompt, user_prompt, decoded_image


def _prepare_parsed_request(payload: ChatRequest) -> ParsedRequest:
    if assets is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if payload.model and payload.model != settings.served_model_name:
        raise HTTPException(status_code=400, detail="requested model is not available")
    system_prompt, user_prompt, image = _coerce_prompts(payload.messages)
    if payload.max_tokens and payload.max_tokens > settings.max_output_tokens:
        raise HTTPException(status_code=400, detail="max_tokens exceeds service limit")
    max_new_tokens = min(payload.max_tokens or settings.default_max_tokens, settings.max_output_tokens)

    messages_for_template: List[Mapping[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    prompt_text = assets.tokenizer.apply_chat_template(
        messages_for_template,
        tokenize=False,
        add_generation_prompt=True,
    )
    return ParsedRequest(
        prompt_text=prompt_text,
        image=image,
        max_new_tokens=max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stream=payload.stream,
    )


def _build_generation_inputs(parsed: ParsedRequest) -> tuple[dict[str, Any], int]:
    inputs = assets.processor(
        text=[parsed.prompt_text],
        images=[parsed.image],
        return_tensors="pt",
    )
    inputs = inputs.to(assets.device)
    if not hasattr(inputs, "pixel_values"):
        raise HTTPException(status_code=400, detail="image inputs missing pixel values")
    sequence_length = int(inputs.input_ids.shape[1])
    return inputs, sequence_length


def _generate_caption(parsed: ParsedRequest) -> str:
    inputs, sequence_length = _build_generation_inputs(parsed)
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values,
        "max_new_tokens": parsed.max_new_tokens,
        "do_sample": parsed.temperature > 0.0,
        "temperature": parsed.temperature if parsed.temperature > 0.0 else None,
        "top_p": parsed.top_p,
        "eos_token_id": assets.tokenizer.eos_token_id,
        "pad_token_id": assets.tokenizer.pad_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.inference_mode():
        generated_ids = assets.model.generate(**generation_kwargs)
    generated_sequence = generated_ids[0][sequence_length:]
    caption = assets.tokenizer.decode(generated_sequence, skip_special_tokens=True)
    return caption.strip()


def _run_stream_generate(parsed: ParsedRequest, streamer: TextIteratorStreamer) -> None:
    inputs, _ = _build_generation_inputs(parsed)
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values,
        "max_new_tokens": parsed.max_new_tokens,
        "do_sample": parsed.temperature > 0.0,
        "temperature": parsed.temperature if parsed.temperature > 0.0 else None,
        "top_p": parsed.top_p,
        "eos_token_id": assets.tokenizer.eos_token_id,
        "pad_token_id": assets.tokenizer.pad_token_id,
        "streamer": streamer,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
    with torch.inference_mode():
        assets.model.generate(**generation_kwargs)


async def _stream_response(parsed: ParsedRequest, completion_id: str, created: int):
    streamer = TextIteratorStreamer(
        assets.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    worker = threading.Thread(target=_run_stream_generate, args=(parsed, streamer), daemon=True)
    worker.start()

    try:
        for text in streamer:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": settings.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data, separators=(',', ':'))}\n\n"
    finally:
        worker.join(timeout=1.0)
    done = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": settings.served_model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _record_metrics(endpoint: str, status: str, duration: float) -> None:
    REQUEST_COUNTER.labels(protocol="http", endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(protocol="http", endpoint=endpoint).observe(duration)


app = FastAPI()


@app.get("/v1/health/live", dependencies=[Depends(_require_http_auth)])
async def live() -> Mapping[str, bool]:
    return {"live": True}


@app.get("/v1/health/ready", dependencies=[Depends(_require_http_auth)])
async def ready() -> Mapping[str, Any]:
    return {"ready": model_state.loaded, "error": model_state.last_error}


@app.get("/v1/models", dependencies=[Depends(_require_http_auth)])
async def models() -> Mapping[str, Any]:
    return {"data": [{"id": settings.served_model_name, "object": "model"}]}


@app.get("/v1/metadata", dependencies=[Depends(_require_http_auth)])
async def metadata() -> Mapping[str, Any]:
    return {
        "id": settings.served_model_name,
        "model_id": settings.model_id,
        "version": settings.model_version,
        "max_batch_size": settings.max_batch_size,
        "max_output_tokens": settings.max_output_tokens,
    }


@app.get("/metrics", dependencies=[Depends(_require_http_auth)])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", dependencies=[Depends(_require_http_auth)])
async def chat_completions(request: Request) -> Response:
    started = time.perf_counter()
    status_label = "200"
    INFLIGHT_GAUGE.labels(protocol="http", endpoint="/v1/chat/completions").inc()
    try:
        payload = ChatRequest.model_validate(await request.json())
        if not model_state.loaded or assets is None:
            raise HTTPException(status_code=503, detail="model not ready")
        parsed = _prepare_parsed_request(payload)
        completion_id = f"chatcmpl-{os.urandom(6).hex()}"
        created = int(time.time())

        span_cm = TRACER.start_as_current_span("chat_completions") if TRACER else nullcontext()
        rate_cm = async_rate_limiter.limit()
        async with span_cm, rate_cm:
            loop = asyncio.get_running_loop()
            if parsed.stream:
                return StreamingResponse(
                    _stream_response(parsed, completion_id, created),
                    media_type="text/event-stream",
                )
            caption = await loop.run_in_executor(None, _generate_caption, parsed)
        response = ChatResponse(
            id=completion_id,
            object="chat.completion",
            created=created,
            model=settings.served_model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message={"role": "assistant", "content": caption},
                    finish_reason="stop",
                )
            ],
        )
        return JSONResponse(content=response.model_dump())
    except HTTPException as exc:
        status_label = str(exc.status_code)
        raise
    except ValidationError as exc:
        status_label = "400"
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to logs
        logger.exception("Unhandled error during chat completion")
        status_label = "500"
        raise HTTPException(status_code=500, detail="internal server error") from exc
    finally:
        duration = time.perf_counter() - started
        INFLIGHT_GAUGE.labels(protocol="http", endpoint="/v1/chat/completions").dec()
        _record_metrics("/v1/chat/completions", status_label, duration)


@app.get("/", dependencies=[Depends(_require_http_auth)])
async def root() -> Mapping[str, str]:
    return {"service": settings.served_model_name, "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=settings.http_port,
        reload=False,
    )
