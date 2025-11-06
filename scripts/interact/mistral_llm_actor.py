#!/usr/bin/env python3
"""
Ray actor wrapper for serving Mistral 7B Instruct on vLLM and proxying requests.

This script:
- Defines a GPU-enabled Ray actor `MistralLLMServer` that launches a vLLM OpenAI server
  as a subprocess and exposes simple chat methods that work from the driver.
- Provides a Click-based CLI to deploy the actor to a Ray cluster and run a sample query.

Notes:
- Requires a Ray GPU worker with at least 1 visible GPU.
- Installs vLLM nightly at runtime inside the actor if it's not already present.
- Uses the OpenAI-compatible HTTP server started by `vllm serve` inside the actor container.
- The actor proxies queries locally (127.0.0.1) to that server and returns the result to the driver.

Example:
  python scripts/interact/mistral_llm_actor.py deploy --address ray://ray-head:10001 \
    --model mistralai/Mistral-7B-Instruct-v0.3 --dtype float16 --run-sample
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from typing import Any, Dict, Optional, List

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


def _wait_http_ready(url: str, timeout_s: float = 1200.0, interval_s: float = 1.0) -> None:
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
class MistralLLMServer:
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        host: str = "0.0.0.0",
        port: int = 8002,
        dtype: Optional[str] = "float16",  # common default for this model
        quantization: Optional[str] = None,  # e.g., "awq", "gptq", "modelopt_fp4"
        cache_dir: Optional[str] = "/workspace/models",
        readiness_timeout_s: float = 1200.0,
        extra_args: Optional[List[str]] = None,
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
        ]
        if self._dtype:
            cmd += ["--dtype", self._dtype]
        if self._quant:
            cmd += ["--quantization", self._quant]
        # Constrain to the single GPU assigned by Ray and use the mounted cache
        cmd += ["--tensor-parallel-size", "1"]
        cmd += ["--download-dir", self._cache_dir]
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
        # Send logs to worker stdout/stderr to prevent PIPE buffer blocking
        self._p = subprocess.Popen(
            cmd,
            env=env,
            stdout=None,
            stderr=None,
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
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def health(self) -> Dict[str, Any]:
        self._ensure_alive()
        r = requests.get(f"http://127.0.0.1:{self._port}/v1/models", timeout=5)
        return {"status_code": r.status_code, "body": r.json() if r.ok else r.text}

    def chat(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
        model_override: Optional[str] = None,
    ) -> str:
        """Simple text-only chat for Mistral Instruct."""
        self._ensure_alive()
        from openai import OpenAI  # lazy import; installed in env

        base_url = f"http://127.0.0.1:{self._port}/v1"
        client = OpenAI(base_url=base_url, api_key="null")
        model_name = model_override or self._model

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
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
    """Manage the Mistral 7B Instruct vLLM actor on a Ray cluster."""
    pass


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="mistral_llm", show_default=True, help="Detached actor name")
@click.option("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", show_default=True)
@click.option("--dtype", type=str, default="float16", show_default=True, help="e.g., float16 or bfloat16")
@click.option("--quantization", type=str, default=None, help="e.g., awq, gptq, modelopt_fp4")
@click.option("--host", type=str, default="0.0.0.0", show_default=True)
@click.option("--port", type=int, default=8002, show_default=True)
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
    "--run-sample", is_flag=True, default=False, show_default=True, help="Run a sample text chat after deployment"
)
@click.option(
    "--sample-prompt", type=str, default="List three practical use cases for Mistral-7B-Instruct.", show_default=True
)
def deploy(
    address: Optional[str],
    namespace: str,
    actor_name: str,
    model: str,
    dtype: Optional[str],
    quantization: Optional[str],
    host: str,
    port: int,
    cache_dir: str,
    readiness_timeout_s: float,
    run_sample: bool,
    sample_prompt: str,
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
        actor = MistralLLMServer.options(name=actor_name, lifetime="detached").remote(
            model=model,
            host=host,
            port=port,
            dtype=dtype,
            quantization=quantization,
            cache_dir=cache_dir,
            readiness_timeout_s=readiness_timeout_s,
        )
        # Touch actor to force construction
        info = ray.get(actor.info.remote())
        print("Actor info:", info)

    if run_sample:
        print("Running sample text chat...")
        out = ray.get(actor.chat.remote(sample_prompt))
        print("\nResponse:\n", out)


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="mistral_llm", show_default=True)
@click.option("--prompt", type=str, required=True)
def query(address: Optional[str], namespace: str, actor_name: str, prompt: str) -> None:
    """Send a single text chat to the deployed actor and print the response."""
    resolved = address or os.environ.get("RAY_ADDRESS", "ray://ray-head:10001")
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, namespace=namespace, ignore_reinit_error=True)
    actor = ray.get_actor(actor_name, namespace=namespace)
    out = ray.get(actor.chat.remote(prompt))
    print(out)


@cli.command()
@click.option(
    "--address",
    type=str,
    default=None,
    help="Ray address (e.g., ray://ray-head:10001). Defaults to RAY_ADDRESS or ray://ray-head:10001",
)
@click.option("--namespace", type=str, default="default", show_default=True, help="Ray namespace for the actor")
@click.option("--actor-name", type=str, default="mistral_llm", show_default=True)
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
