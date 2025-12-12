# paddleocr (open replacement)

Reverse-engineered drop-in for the private `nvcr.io/nim/baidu/paddleocr` NIM using the open-source [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) pipeline.

## What nv-ingest expects
- HTTP endpoint `${OCR_HTTP_ENDPOINT:-http://ocr:8000/v1/infer}` (no gRPC surface here).
- Request body: `{"input": [{"type": "image_url", "url": "data:image/png;base64,<...>"}, ...]}` with up to `NIM_TRITON_MAX_BATCH_SIZE` images (defaults to 2 in this shim). URLs may be `data:` URIs, http/https links, or local file paths.
- Response: `{"data": [{"text_detections": [{"bounding_box": {"points": [{"x": <float>, "y": <float>}, ...], "type": "quadrilateral"}, "text_prediction": {"text": "...", "confidence": 1.0}}, ...]}]}`. Bounding boxes are normalized to `[0,1]` with four points per detection.
- Health endpoints: `/v1/health/live` and `/v1/health/ready` return HTTP 200. Metadata: `/v1/metadata` exposes `modelInfo[0].shortName = "paddleocr"`.

## Open implementation in this directory
- `server.py`: FastAPI server that wraps `PaddleOCRVL` from `paddleocr`. Images are decoded to BGR numpy arrays, run through the PaddleOCR-VL doc parser, and emitted as `text_detections` with normalized quadrilateral boxes and confidences (fixed at 1.0 because the public pipeline does not expose per-block scores).
- `requirements.txt`: Runtime deps for the server. Install PaddlePaddle GPU separately for your CUDA version (per official instructions) before installing these Python deps.

## Setup
1) Install PaddlePaddle + PaddleOCR-VL stack (pick the wheel that matches your CUDA):
```bash
# Example for CUDA 12.6; adjust the index/wheel per https://www.paddlepaddle.org.cn/en/install/quick
python3 -m pip install "paddlepaddle-gpu==3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python3 -m pip install -U "paddleocr[doc-parser]==3.3.2" paddlex==3.3.11 opencv-contrib-python==4.10.0.84
# Some environments require the nightly safetensors wheel referenced in the PaddleOCR-VL model card.
```

2) Install the server deps:
```bash
cd nim/baidu/paddleocr
python3 -m pip install -r requirements.txt
```

3) Run the server:
```bash
OCR_MODEL_NAME=paddle \
OCR_INFER_PROTOCOL=http \
OCR_HTTP_ENDPOINT=http://127.0.0.1:8000/v1/infer \
NIM_TRITON_MAX_BATCH_SIZE=2 \
uvicorn server:app --host 0.0.0.0 --port ${NIM_HTTP_API_PORT:-8000}
```

4) Wire nv-ingest to it:
- Set `OCR_MODEL_NAME=paddle` so nv-ingest selects the paddle interface.
- Set `OCR_INFER_PROTOCOL=http` and `OCR_HTTP_ENDPOINT=http://<host>:8000/v1/infer`.
- Leave `OCR_GRPC_ENDPOINT` empty to avoid gRPC.
- If using Helm, override `nimOperator.paddleocr.expose.service.port` to 8000 and point `envVars.OCR_HTTP_ENDPOINT` at the HTTP URL above.

## Smoke test
```bash
base64_img=$(python3 - <<'PY'
from PIL import Image
import base64, io
img = Image.new('RGB', (200, 80), color='white')
b = io.BytesIO()
img.save(b, format='PNG')
print(base64.b64encode(b.getvalue()).decode())
PY)

curl -X POST http://127.0.0.1:${NIM_HTTP_API_PORT:-8000}/v1/infer \
  -H 'content-type: application/json' \
  -d '{"input":[{"type":"image_url","url":"data:image/png;base64:'"$base64_img"'"}]}'
```
