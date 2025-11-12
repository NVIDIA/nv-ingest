#!/usr/bin/env python3
"""
Ray actor wrapper for serving NVIDIA Nemotron Nano 2 VL on vLLM and proxying requests.

This script:
- Defines a GPU-enabled Ray actor `NemotronVLMServer` that launches a vLLM OpenAI server
  as a subprocess and exposes simple chat methods that work from the driver.
- Provides a Click-based CLI to deploy the actor to a Ray cluster and run a sample query.

Notes:
- Requires a Ray GPU worker with at least 1 visible GPU.
- Installs vLLM nightly at runtime inside the actor if it's not already present.
- Uses the OpenAI-compatible HTTP server started by `vllm serve` inside the actor container.
- The actor proxies queries locally (127.0.0.1) to that server and returns the result to the driver.

Example:
  python scripts/interact/nemo_llm_actor.py deploy --address ray://ray-head:10001 \
    --model nvidia/Nemotron-Nano-12B-v2-VL-BF16 --dtype bfloat16 --video-pruning-rate 0 --run-sample

"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Any, Dict, Optional

import click
import requests
import ray


def _pip_install_if_missing() -> None:
    """Ensure vLLM nightly is installed inside the actor process."""
    try:
        import vllm  # noqa: F401

        return
    except Exception:
        pass
    # Install nightly wheels per vLLM guidance
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "vllm",
        "--extra-index-url",
        "https://wheels.vllm.ai/nightly",
        "--pre",
    ]
    subprocess.check_call(cmd)


def _wait_http_ready(url: str, timeout_s: float = 120.0, interval_s: float = 1.0) -> None:
    start = time.time()
    while True:
        try:
            r = requests.get(url, timeout=3)
            if r.ok:
                return
        except Exception:
            pass
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for {url} to become ready")
        time.sleep(interval_s)


@ray.remote(num_gpus=1)
class NemotronVLMServer:
    def __init__(
        self,
        model: str = "nvidia/Nemotron-Nano-12B-v2-VL-BF16",
        host: str = "0.0.0.0",
        port: int = 8000,
        dtype: Optional[str] = "bfloat16",  # e.g., bfloat16 for BF16 variant
        quantization: Optional[str] = None,  # e.g., "modelopt" (FP8) or "modelopt_fp4" (FP4)
        video_pruning_rate: float = 0.0,
        cache_dir: Optional[str] = "/workspace/models",
        readiness_timeout_s: float = 1800.0,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        """
        Launch vLLM OpenAI server for the given model inside this actor process.
        """
        _pip_install_if_missing()

        self._model = model
        self._host = host
        self._port = int(port)
        self._dtype = dtype
        self._quant = quantization
        self._vpr = float(video_pruning_rate)
        self._p: Optional[subprocess.Popen] = None
        self._cache_dir = cache_dir or "/workspace/models"
        self._ready_timeout = float(readiness_timeout_s)

        # Prepare cache directories for HF/vLLM
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self._cache_dir, "hf"), exist_ok=True)
            os.makedirs(os.path.join(self._cache_dir, "transformers"), exist_ok=True)
            os.makedirs(os.path.join(self._cache_dir, "vllm"), exist_ok=True)
        except Exception:
            # best-effort
            pass

        # Build vLLM serve command
        cmd = [
            "vllm",
            "serve",
            self._model,
            "--trust-remote-code",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--video-pruning-rate",
            str(self._vpr),
        ]
        if self._dtype:
            cmd += ["--dtype", self._dtype]
        if self._quant:
            cmd += ["--quantization", self._quant]
        if extra_args:
            cmd += list(extra_args)

        # Launch server
        env = os.environ.copy()
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        # Point caches to persistent location to avoid repeated downloads
        env.setdefault("HF_HOME", os.path.join(self._cache_dir, "hf"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(self._cache_dir, "hf", "hub"))
        env.setdefault("TRANSFORMERS_CACHE", os.path.join(self._cache_dir, "transformers"))
        env.setdefault("VLLM_CACHE_DIR", os.path.join(self._cache_dir, "vllm"))
        self._p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for readiness
        base_url = f"http://127.0.0.1:{self._port}"
        _wait_http_ready(f"{base_url}/v1/models", timeout_s=self._ready_timeout)

    def _ensure_alive(self) -> None:
        if self._p is None:
            raise RuntimeError("Server process not started")
        ret = self._p.poll()
        if ret is not None:
            # Read a bit of output for diagnostics
            try:
                tail = self._p.stdout.read() if self._p.stdout else ""
            except Exception:
                tail = ""
            raise RuntimeError(f"vLLM server exited with code {ret}. Output: {tail[-2048:]}")

    def info(self) -> Dict[str, Any]:
        self._ensure_alive()
        return {
            "model": self._model,
            "host": self._host,
            "port": self._port,
            "dtype": self._dtype,
            "quantization": self._quant,
            "video_pruning_rate": self._vpr,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def health(self) -> Dict[str, Any]:
        self._ensure_alive()
        r = requests.get(f"http://127.0.0.1:{self._port}/v1/models", timeout=5)
        return {"status_code": r.status_code, "body": r.json() if r.ok else r.text}

    def chat_image(
        self,
        image_url: str,
        text: str,
        system: str = "/no_think",
        temperature: float = 0.0,
        max_tokens: int = 512,
        model_override: Optional[str] = None,
    ) -> str:
        """Simple multimodal chat with one image URL and a text prompt."""
        self._ensure_alive()
        from openai import OpenAI  # lazy import; installed in env

        base_url = f"http://127.0.0.1:{self._port}/v1"
        client = OpenAI(base_url=base_url, api_key="null")
        model_name = model_override or self._model

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def stop(self) -> None:
        if self._p is not None:
            try:
                self._p.terminate()
                try:
                    self._p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._p.kill()
            finally:
                self._p = None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Manage the Nemotron Nano 2 VL vLLM actor on a Ray cluster."""
    pass


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="nemo_vlm", show_default=True, help="Detached actor name")
@click.option("--model", type=str, default="nvidia/Nemotron-Nano-12B-v2-VL-BF16", show_default=True)
@click.option("--dtype", type=str, default="bfloat16", show_default=True, help="e.g., bfloat16 for BF16 model")
@click.option("--quantization", type=str, default=None, help="e.g., modelopt (FP8) or modelopt_fp4 (FP4)")
@click.option("--video-pruning-rate", type=float, default=0.0, show_default=True)
@click.option("--host", type=str, default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option(
    "--cache-dir",
    type=str,
    default="/workspace/models",
    show_default=True,
    help="Persistent cache directory inside worker",
)
@click.option(
    "--readiness-timeout",
    "readiness_timeout_s",
    type=float,
    default=1200.0,
    show_default=True,
    help="Seconds to wait for vLLM server readiness",
)
@click.option(
    "--run-sample", is_flag=True, default=False, show_default=True, help="Run a sample image+text chat after deployment"
)
@click.option(
    "--sample-image-url",
    type=str,
    default="https://blogs.nvidia.com/wp-content/uploads/2025/08/gamescom-g-assist-nv-blog-1280x680-1.jpg",
)
@click.option("--sample-text", type=str, default="Give me 3 interesting facts about this image.")
def deploy(
    address: Optional[str],
    namespace: str,
    actor_name: str,
    model: str,
    dtype: Optional[str],
    quantization: Optional[str],
    video_pruning_rate: float,
    host: str,
    port: int,
    cache_dir: str,
    readiness_timeout_s: float,
    run_sample: bool,
    sample_image_url: str,
    sample_text: str,
) -> None:
    """Deploy the actor and optionally run a sample query."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)

    # Get or create the detached actor
    try:
        actor = ray.get_actor(actor_name, namespace=namespace)
        print(f"Found existing actor: {actor_name}")
    except Exception:
        print(f"Creating actor: {actor_name}")
        actor = NemotronVLMServer.options(name=actor_name, lifetime="detached").remote(
            model=model,
            host=host,
            port=port,
            dtype=dtype,
            quantization=quantization,
            video_pruning_rate=video_pruning_rate,
            cache_dir=cache_dir,
            readiness_timeout_s=readiness_timeout_s,
        )
        # Touch actor to force construction
        info = ray.get(actor.info.remote())
        print("Actor info:", info)

    if run_sample:
        print("Running sample image+text chat...")
        out = ray.get(actor.chat_image.remote(sample_image_url, sample_text))
        print("\nResponse:\n", out)


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="nemo_vlm", show_default=True)
@click.option("--image-url", type=str, required=True)
@click.option("--text", type=str, required=True)
def query(address: Optional[str], namespace: str, actor_name: str, image_url: str, text: str) -> None:
    """Send a single image+text chat to the deployed actor and print the response."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)
    actor = ray.get_actor(actor_name, namespace=namespace)
    out = ray.get(actor.chat_image.remote(image_url, text))
    print(out)


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="nemo_vlm", show_default=True)
def stop(address: Optional[str], namespace: str, actor_name: str) -> None:
    """Stop the vLLM server process inside the actor (actor remains)."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)
    actor = ray.get_actor(actor_name, namespace=namespace)
    ray.get(actor.stop.remote())
    print("Stopped vLLM server process.")


if __name__ == "__main__":
    cli()
