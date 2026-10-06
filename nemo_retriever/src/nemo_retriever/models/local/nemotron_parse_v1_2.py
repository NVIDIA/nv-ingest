# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision
from nemo_retriever.models.model import BaseModel, ModelRunMode

# Type alias for all supported single-image input formats.
ImageInput = Union[torch.Tensor, np.ndarray, Image.Image, str, Path]


# ---------------------------------------------------------------------------
# vLLM processor bug workaround
# ---------------------------------------------------------------------------
# vLLM's bundled NemotronParseProcessor.__call__ passes add_special_tokens=False
# explicitly to the tokenizer AND also forwards it via **kwargs from the vLLM
# pipeline, causing a duplicate keyword argument TypeError.  We monkey-patch
# the processor class at import time to pop the conflicting kwarg.

_VLLM_PROCESSOR_PATCHED = False


def _patch_vllm_nemotron_parse_processor() -> None:
    """Fix duplicate-kwarg bug in vLLM's NemotronParseProcessor.__call__."""
    global _VLLM_PROCESSOR_PATCHED
    if _VLLM_PROCESSOR_PATCHED:
        return

    try:
        from vllm.model_executor.models.nemotron_parse import NemotronParseProcessor
    except ImportError:
        return

    _orig_call = NemotronParseProcessor.__call__

    def _fixed_call(self, text=None, images=None, return_tensors=None, **kwargs):
        kwargs.pop("add_special_tokens", None)
        return _orig_call(self, text=text, images=images, return_tensors=return_tensors, **kwargs)

    NemotronParseProcessor.__call__ = _fixed_call
    _VLLM_PROCESSOR_PATCHED = True


def _run_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    try:
        loop.run_forever()
    finally:
        loop.close()


def _stop_event_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join()


_BATCH_TASK_NAME = "nemotron-parse-batch"


async def _cancel_requests_and_shut_down(engine: Any) -> None:
    # vLLM's own shutdown stops the engine core before cancelling its output handler, which then logs
    # EngineDeadError; cancelling in-flight batches and the handler first lets the engine stop quietly.
    tasks = [task for task in asyncio.all_tasks() if task.get_name() == _BATCH_TASK_NAME]
    if (handler := getattr(engine, "output_handler", None)) is not None:
        tasks.append(handler)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    engine.shutdown()


def _shut_down(engine: Any, loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    if not thread.is_alive():
        return
    if threading.current_thread() is thread:
        loop.create_task(_cancel_requests_and_shut_down(engine)).add_done_callback(lambda _task: loop.stop())
        return
    try:
        asyncio.run_coroutine_threadsafe(_cancel_requests_and_shut_down(engine), loop).result()
    finally:
        _stop_event_loop(loop, thread)


class _AsyncEngine:
    """One vLLM ``AsyncLLM`` on a private event-loop thread, shared by concurrent callers.

    Every caller's pages become requests to the same scheduler, so one batch's
    slowest pages no longer hold the GPU while the next batch waits. The engine
    and its thread stop once: on :meth:`close`, when this object is collected,
    or at interpreter exit.
    """

    def __init__(self, engine_kwargs: dict[str, Any]) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=_run_event_loop, args=(self._loop,), name="nemotron-parse-engine", daemon=True
        )
        self._thread.start()

        async def create() -> Any:
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))

        try:
            self._engine = asyncio.run_coroutine_threadsafe(create(), self._loop).result()
        except BaseException:
            _stop_event_loop(self._loop, self._thread)
            raise
        # Created after the engine so that, at interpreter exit, it runs before vLLM's own finalizers.
        self._finalizer = weakref.finalize(self, _shut_down, self._engine, self._loop, self._thread)

    def close(self) -> None:
        """Cancel in-flight requests, shut down the engine, and stop its thread; later calls do nothing."""
        self._finalizer()

    def generate(self, prompts: Sequence[Any], sampling_params: Any) -> List[Any]:
        """Return each prompt's final output in input order, like ``LLM.generate``.

        The first failure cancels the batch's other requests, which vLLM aborts,
        and then propagates unchanged.
        """

        async def final_output(prompt: Any) -> Any:
            output = None
            async for output in self._engine.generate(prompt, sampling_params, uuid.uuid4().hex):
                pass
            return output

        async def generate_all() -> List[Any]:
            asyncio.current_task().set_name(_BATCH_TASK_NAME)
            tasks = [asyncio.ensure_future(final_output(prompt)) for prompt in prompts]
            try:
                return list(await asyncio.gather(*tasks))
            except Exception:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return asyncio.run_coroutine_threadsafe(generate_all(), self._loop).result()


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class NemotronParseV12(BaseModel):
    """
    NVIDIA Nemotron Parse v1.2 local wrapper backed by vLLM.

    This wrapper loads ``nvidia/NVIDIA-Nemotron-Parse-v1.2`` via vLLM's offline
    ``LLM`` engine for image-to-structured-text generation (document parsing).
    vLLM handles KV-cache management, continuous batching, and GPU scheduling
    internally, avoiding the transformers cache-API incompatibility that affects
    the HuggingFace ``trust_remote_code`` model code with transformers >= 4.52.

    With ``async_engine=True`` it uses vLLM's ``AsyncLLM`` instead, with the same
    model and sampling settings, so several threads can submit batches to one
    engine at once. Use it only when the caller runs batches concurrently.
    """

    _DEFAULT_TASK_PROMPT: str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

    def __init__(
        self,
        model_path: str = "nvidia/NVIDIA-Nemotron-Parse-v1.2",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        task_prompt: str = _DEFAULT_TASK_PROMPT,
        gpu_memory_utilization: float = 0.8,
        max_num_seqs: int = 64,
        max_tokens: int = 9000,
        async_engine: bool = False,
    ) -> None:
        super().__init__()

        from nemo_retriever.models.inference.vllm import apply_vllm_startup_defaults

        apply_vllm_startup_defaults()
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Local Nemotron Parse requires vLLM. " 'Install with: pip install "nemo-retriever[vllm]"'
            ) from e

        _patch_vllm_nemotron_parse_processor()

        self._model_path = model_path
        self._task_prompt = task_prompt
        self._max_tokens = max_tokens

        if device is not None:
            import os

            dev_id = device.split(":")[-1] if ":" in device else device
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_id

        configure_global_hf_cache_base(hf_cache_dir)
        revision = get_hf_revision(model_path)

        engine_kwargs: dict[str, Any] = dict(
            model=model_path,
            revision=revision,
            trust_remote_code=True,
            dtype="bfloat16",
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=gpu_memory_utilization,
        )
        sampling_kwargs: dict[str, Any] = dict(
            temperature=0,
            top_k=1,
            repetition_penalty=1.1,
            max_tokens=self._max_tokens,
            skip_special_tokens=False,
        )
        if async_engine:
            from vllm.sampling_params import RequestOutputKind

            # The offline LLM applies these engine defaults itself; AsyncEngineArgs does not.
            self._llm = _AsyncEngine({**engine_kwargs, "seed": 0, "disable_log_stats": True})
            sampling_kwargs["output_kind"] = RequestOutputKind.FINAL_ONLY
        else:
            self._llm = LLM(**engine_kwargs)

        self._sampling_params = SamplingParams(**sampling_kwargs)

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def preprocess(self, input_data: ImageInput) -> Image.Image:
        """Normalize supported input formats to an RGB PIL image."""
        if isinstance(input_data, Image.Image):
            return input_data.convert("RGB")

        if isinstance(input_data, (str, Path)):
            return Image.open(Path(input_data)).convert("RGB")

        if isinstance(input_data, torch.Tensor):
            x = input_data.detach().cpu()
            if x.ndim == 4:
                if int(x.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 tensor, got shape {tuple(x.shape)}")
                x = x[0]
            if x.ndim != 3:
                raise ValueError(f"Expected CHW/HWC tensor, got shape {tuple(x.shape)}")
            if int(x.shape[0]) in (1, 3):
                x = x.permute(1, 2, 0).contiguous()
            if x.dtype.is_floating_point:
                max_v = float(x.max().item()) if x.numel() else 1.0
                if max_v <= 1.5:
                    x = x * 255.0
            arr = x.clamp(0, 255).to(torch.uint8).numpy()
            return Image.fromarray(arr).convert("RGB")

        if isinstance(input_data, np.ndarray):
            arr = input_data
            if arr.ndim == 4:
                if int(arr.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 array, got shape {arr.shape}")
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected HWC/CHW array, got shape {arr.shape}")
            if int(arr.shape[0]) in (1, 3) and int(arr.shape[-1]) not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                max_v = float(arr.max()) if arr.size else 1.0
                if max_v <= 1.5:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise TypeError(f"Unsupported input type for Nemotron Parse: {type(input_data)!r}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def invoke(
        self,
        input_data: ImageInput,
        task_prompt: Optional[str] = None,
    ) -> str:
        """Run Nemotron Parse on a single image and return the decoded text."""
        return self.invoke_batch([input_data], task_prompt=task_prompt)[0]

    def invoke_batch(
        self,
        inputs: Sequence[ImageInput],
        task_prompt: Optional[str] = None,
    ) -> List[str]:
        """Run Nemotron Parse on a batch of images via vLLM.

        vLLM handles continuous batching and GPU scheduling internally,
        making this significantly faster than sequential single-image calls
        for large batches.
        """
        return [text.strip() for text, _ in self.invoke_batch_with_finish_reasons(inputs, task_prompt=task_prompt)]

    def invoke_batch_with_finish_reasons(
        self,
        inputs: Sequence[ImageInput],
        task_prompt: Optional[str] = None,
    ) -> List[tuple[str, str]]:
        """Run a batch and return each unstripped completion with its vLLM finish reason.

        A finish reason other than ``"stop"``, such as ``"length"``, means the
        generation ended before the model completed the page.
        Errors from image preprocessing or vLLM generation propagate unchanged.

        Args:
            inputs: Sequence of PIL images, image paths (strings or ``Path``
                objects), tensors, or NumPy arrays, normalized to RGB images.
                Tensors and arrays must be CHW or HWC, optionally with a leading
                batch dimension of size 1.
            task_prompt: Decoder prompt for every image. ``None`` or an empty
                string uses the prompt configured on this model.

        Returns:
            A list of ``(text, finish_reason)`` tuples in input order. Text keeps
            its original whitespace. A missing or empty finish reason becomes
            ``"unknown"``.

        Raises:
            TypeError: An input type is unsupported.
            ValueError: A tensor or array has an unsupported shape.
            OSError: An image file cannot be opened or decoded.
            IndexError: vLLM returns a request without a completion.
            RuntimeError: The model was closed.
        """
        if self._llm is None:
            raise RuntimeError("This Nemotron Parse model was closed; create a new NemotronParseV12 to run it again.")
        prompt = task_prompt or self._task_prompt
        prompts = [
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {"image": self.preprocess(img)},
                },
                "decoder_prompt": prompt,
            }
            for img in inputs
        ]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [(out.outputs[0].text, str(out.outputs[0].finish_reason or "unknown")) for out in outputs]

    def __call__(
        self,
        input_data: ImageInput,
        task_prompt: Optional[str] = None,
    ) -> str:
        return self.invoke(input_data, task_prompt=task_prompt)

    def close(self) -> None:
        """Release the vLLM engine and the GPU memory it holds; later calls do nothing.

        The async engine stops at once. vLLM's offline engine has no shutdown API, so
        this drops the model's reference and vLLM's own finalizer stops its engine
        process. The model cannot run afterwards.
        """
        engine, self._llm = getattr(self, "_llm", None), None
        if isinstance(engine, _AsyncEngine):
            engine.close()

    # ------------------------------------------------------------------
    # BaseModel abstract interface
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return "NVIDIA-Nemotron-Parse-v1.2"

    @property
    def model_type(self) -> str:
        return "document-parse"

    @property
    def model_runmode(self) -> ModelRunMode:
        return "local"

    @property
    def input(self) -> Any:
        return {
            "type": "image",
            "format": "RGB",
            "supported_inputs": ["PIL.Image", "path", "torch.Tensor", "np.ndarray"],
            "description": "Document image for parsing into markdown text with structural tags.",
        }

    @property
    def output(self) -> Any:
        return {
            "type": "text",
            "format": "string",
            "description": "Generated structured parse text from Nemotron Parse v1.2.",
        }

    @property
    def input_batch_size(self) -> int:
        return 64
