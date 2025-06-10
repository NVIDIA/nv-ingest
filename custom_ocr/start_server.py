import base64
import json
import logging
import time
from typing import List

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scene_text_inference import SceneTextAPI
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import decode_image
from torchvision.io import read_file

# Set the root logger to WARNING so that standard INFO logs are not printed.
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Setup a dedicated performance logger.
perf_logger = logging.getLogger("perf")
perf_logger.setLevel(logging.INFO)
# Ensure that the performance logger outputs messages in a simple format.
if not perf_logger.hasHandlers():
    perf_handler = logging.StreamHandler()
    perf_handler.setFormatter(logging.Formatter("%(message)s"))
    perf_logger.addHandler(perf_handler)

# Initialize your inference API.
scene_text = SceneTextAPI("rtx_v2")
IDLE_BYTES = 1024 * 1024 * 1024 * 20  # 20GB
app = FastAPI()



# Custom JSON encoder to support torch.Tensor and objects with __dict__
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


# Pydantic model for a single inference request.
class InferenceRequest(BaseModel):
    type: str  # Either "image_url" for base64 or "image_path" for a file path
    url: str


# Pydantic models for a batch inference request.
class InputItem(BaseModel):
    type: str  # "image_url" or "image_path"
    url: str


class BatchRequest(BaseModel):
    input: List[InputItem]


def load_image_as_tensor(image_path):
    """
    Loads an image from disk, decodes it, and returns a tensor.
    """
    image_bytes = read_file(image_path)
    tensor = decode_image(image_bytes, mode=ImageReadMode.RGB)
    return tensor


def _maybe_flush():
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    idle = reserved - allocated
    if idle <= IDLE_BYTES:
        return                             # cache small enough – skip
    torch.cuda.synchronize()               # finish in-flight kernels
    torch.cuda.empty_cache()               # drop idle blocks
    torch.cuda.ipc_collect()               # clear inter-proc handles


def decode_base64_image_torch(data_url: str) -> torch.Tensor:
    """
    Decode a base64 encoded image into a torch tensor.
    """
    if not data_url.startswith("data:image/"):
        raise ValueError("Invalid base64 image URL.")
    _, encoded_data = data_url.split(",", 1)
    try:
        image_bytes = base64.b64decode(encoded_data)
    except Exception as e:
        raise ValueError("Invalid base64 encoding.") from e

    writable_buf = bytearray(image_bytes)
    image_tensor = decode_image(torch.frombuffer(writable_buf, dtype=torch.uint8), mode=ImageReadMode.RGB)
    return image_tensor


# Modified Dataset to handle both base64 image URLs and file paths.
class ImageDataset(Dataset):
    def __init__(self, items: List[InputItem]):
        """
        Args:
            items (list): List of InputItem objects containing a type and URL.
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if item.type == "image_url":
            tensor = decode_base64_image_torch(item.url)
        elif item.type == "image_path":
            tensor = load_image_as_tensor(item.url)
        else:
            raise ValueError(f"Unsupported type: {item.type}")
        return tensor



@app.post("/infer_single")
def infer_single(single_item: InferenceRequest):
    try:
        data_loading_start = time.time()
        if single_item.type == "image_url":
            tensor = decode_base64_image_torch(single_item.url)
        elif single_item.type == "image_path":
            tensor = load_image_as_tensor(single_item.url)
        else:
            return JSONResponse(content={"error": f"Unsupported type: {single_item.type}"})
        data_loading_end = time.time()
        data_loading_time = data_loading_end - data_loading_start

        inference_start = time.time()
        result = scene_text.infer(tensor)
        inference_end = time.time()
        inference_time = inference_end - inference_start

        _maybe_flush()
        # Log performance metrics using the performance logger.
        perf_logger.info(
            f"Single inference - Data Loading time: {data_loading_time:.4f} s, Inference time: {inference_time:.4f} s"
        )
        json_result = json.dumps(result, cls=CustomEncoder)
        return JSONResponse(content=json.loads(json_result))
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.post("/infer")
def infer(batch: BatchRequest):
    try:
        overall_start = time.time()
        dataset = ImageDataset(batch.input)
        if len(dataset) >= 32:
            num_workers = min(4, len(dataset))
            dataloader = DataLoader(
                dataset,
                batch_size=1,  # keep batch‑size‑1 loading for simplicity
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=min(3, len(dataset) // num_workers),
            )
        else:
            dataloader = dataset
        outputs = []
        total_inference_time = 0.0
        for image_batch in dataloader:
            image_batch = image_batch.squeeze()
            image_batch = image_batch.to("cuda", non_blocking=True).to(torch.float32) / 255.0
            inference_start = time.time()
            result = scene_text.infer(image_batch)
            inference_end = time.time()
            total_inference_time += inference_end - inference_start

            outputs.extend(result if isinstance(result, list) else [result])

        overall_end = time.time()
        total_time = overall_end - overall_start
        data_loading_time = total_time - total_inference_time  # rough split

        perf_logger.info(
            f"Inference - Total time: {total_time:.4f}s | "
            f"Request Size: {len(dataset)} | "
            f"Data Loading: {data_loading_time:.4f}s | "
            f"Inference: {total_inference_time:.4f}s"
        )

        _maybe_flush()
        return JSONResponse(content=json.loads(json.dumps(outputs, cls=CustomEncoder)))
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
