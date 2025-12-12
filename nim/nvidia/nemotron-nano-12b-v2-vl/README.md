Nemotron Nano 12B v2 VL (open)
===============================

Reverse-engineered notes for `nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl` (VLM captioning) using the open `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` release.

What nv-ingest expects
----------------------
- Endpoint: HTTP only; defaults to `VLM_CAPTION_ENDPOINT=http://vlm:8000/v1/chat/completions`. No gRPC path is exercised in the code.
- Model name header: `model` field set from `VLM_CAPTION_MODEL_NAME` (default `nvidia/nemotron-nano-12b-v2-vl`). A Bearer token is sent if `NGC_API_KEY`/`NVIDIA_API_KEY` is set.
- Prompt wiring: system prompt defaults to `/no_think`; user prompt defaults to `Caption the content of this image:`. Both are sent as OpenAI-style chat messages.
- Input shape (per request): one user message per image containing `{type:"text",text:"<prompt>"}` and `{type:"image_url",image_url:{url:"data:image/png;base64,<...>"}}`. Images are base64 PNG data URLs; the client pre-scales them to ~180k chars max.
- Output contract: JSON with `choices` array; each choice must include `message.content` (string caption). Only `choices[*].message.content` is read.
- Batching: with the HTTP path, the client falls back to batch size 1; expect one image per request.

Request/response example
------------------------
```
POST /v1/chat/completions
Authorization: Bearer $NGC_API_KEY
Content-Type: application/json

{
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "/no_think"}]},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Caption the content of this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64_png>"}}
      ]
    }
  ],
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "stream": false
}
```
Response shape (anything with `choices[*].message.content` works):
```
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 0000000000,
  "model": "nvidia/nemotron-nano-12b-v2-vl",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "A caption for the image."},
      "finish_reason": "stop"
    }
  ]
}
```

Running the open model with vLLM
--------------------------------
1) Install dependencies (CUDA toolkit + GPU drivers required):
```
pip install causal_conv1d "transformers>4.53,<4.54" torch timm "mamba-ssm==2.2.5" accelerate open_clip_torch numpy pillow
VLLM_USE_PRECOMPILED=1 pip install "git+https://github.com/vllm-project/vllm.git@main"
```

2) Serve the model with an OpenAI-compatible API that matches the nv-ingest defaults:
```
VLLM_USE_PRECOMPILED=1 vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  --trust-remote-code \
  --dtype bfloat16 \
  --video-pruning-rate 0 \
  --served-model-name nvidia/nemotron-nano-12b-v2-vl \
  --port 8000 \
  --api-key "${NGC_API_KEY:-dummy}"
```
- `--served-model-name` keeps the `model` field compatible with nv-ingest defaults.
- The API key is optional; nv-ingest sends `Authorization: Bearer <key>` if set.

3) Point nv-ingest at the server:
- `VLM_CAPTION_ENDPOINT=http://<host>:8000/v1/chat/completions`
- `VLM_CAPTION_MODEL_NAME=nvidia/nemotron-nano-12b-v2-vl`
- `VLM_CAPTION_PROMPT="Caption the content of this image:"`
- `VLM_CAPTION_SYSTEM_PROMPT="/no_think"`

Notes and parity gaps
---------------------
- The current client does not stream and expects one caption per request; multi-image batching would need a gRPC path that nv-ingest does not use.
- Images arrive as data URLs already resized to the ~180k base64 budget; keep PNG support on the server side.
- If you replace vLLM with another OpenAI-compatible runtime (TRT-LLM, SGLang), keep the same REST surface and `choices[*].message.content` contract.
