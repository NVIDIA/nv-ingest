import asyncio
import base64
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import io

# Global variable to hold the loaded model
# This will be populated during the application's lifespan startup.
model_pipeline = None
# An asyncio.Lock to ensure safe access and reloading of the model
model_lock = asyncio.Lock()

# --- Pydantic Data Models for API Schema ---


class InputItem(BaseModel):
    """Represents a single image item in the batch."""

    type: str = Field("image_url", description="The type of input. Must be 'image_url'.")
    url: str = Field(
        ...,
        description="A Data URL containing the base64-encoded image (e.g., 'data:image/png;base64,...').",
    )


class OCRRequest(BaseModel):
    """Defines the structure for a batch OCR request."""

    input: List[InputItem] = Field(..., description="A list of images to be processed.")
    merge_levels: List[str] = Field(..., description="A list of merge levels, one for each image in the input list.")


class Point(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: List[Point]


class Text(BaseModel):
    text: str
    confidence: float


class TextDetection(BaseModel):
    text_prediction: Text
    bounding_box: Polygon


class OCRResponseItem(BaseModel):
    """Contains the OCR results for a single image from the batch."""

    index: int
    text_detections: List[TextDetection]


class OCRResponse(BaseModel):
    """The final response object containing results for the entire batch."""

    data: List[OCRResponseItem]


# --- Model Loading and Lifespan Management ---


async def load_model():
    """
    Loads or reloads the OCR model. This function is separate
    so it can be called on startup and during error recovery.
    """
    global model_pipeline
    print("Loading OCR model...")
    try:
        from nemo_retriever_ocr.inference.pipeline import NemoRetrieverOCR

        # Initialize the OCR model.
        model_pipeline = NemoRetrieverOCR()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model: {e}")
        model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    # Acquire lock before initial loading to ensure no requests come in until model is ready.
    async with model_lock:
        await load_model()

    yield

    print("Server shutting down...")
    global model_pipeline
    model_pipeline = None


# --- FastAPI Application ---

app = FastAPI(
    title="NVIDIA Nemo Retriever OCR Inference Server",
    description="A server for performing OCR using the nvidia/nemoretriever-ocr-v1 model.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- API Endpoints ---


@app.get("/v1/health/live", tags=["Health"])
async def health_live(response: Response) -> dict:
    """Check if the service is running."""
    return {"live": True}


@app.get("/v1/health/ready", tags=["Health"])
async def health_ready(response: Response) -> dict:
    """Check if the service is ready to receive traffic (i.e., model is loaded)."""
    if model_pipeline is None:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
        return {"ready": False}
    return {"ready": True}


@app.post("/v1/infer", response_model=OCRResponse, tags=["Inference"])
async def perform_inference(request: OCRRequest):
    """
    Performs OCR on a base64-encoded image.
    """
    # Use the lock to ensure exclusive access to the model during inference.
    # If a reload is happening, this will wait until it's finished.
    async with model_lock:
        if model_pipeline is None:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="Model is not loaded or failed to load.",
            )

    if not request.input or len(request.input) != len(request.merge_levels):
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="The 'input' and 'merge_levels' lists must be non-empty and have the same number of elements.",
        )

    images = []
    for item in request.input:
        try:
            header, b64_string = item.url.split(",", 1)
            image_bytes = base64.b64decode(b64_string)
            images.append(io.BytesIO(image_bytes))
        except (ValueError, IndexError):
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Invalid Data URL format or bad base64 content found in one of the input items.",
            )

    all_detections = []
    for idx, (image, merge_level) in enumerate(zip(images, request.merge_levels)):
        try:
            async with model_lock:
                predictions = model_pipeline(image, merge_level=merge_level)

            text_detections = []
            for pred in predictions:
                left, upper, right, lower = (
                    pred["left"],
                    pred["upper"],
                    pred["right"],
                    pred["lower"],
                )

                points = [
                    Point(x=left, y=upper),  # Top-left corner
                    Point(x=right, y=upper),  # Top-right corner
                    Point(x=right, y=lower),  # Bottom-right corner
                    Point(x=left, y=lower),  # Bottom-left corner
                ]

                text_prediction = Text(text=pred["text"], confidence=pred["confidence"])
                bounding_box = Polygon(points=points)

                text_detection = TextDetection(text_prediction=text_prediction, bounding_box=bounding_box)
                text_detections.append(text_detection)

            all_detections.append(OCRResponseItem(index=idx, text_detections=text_detections))

        except Exception as e:
            error_str = str(e)
            # Check for the specific, fatal CUDA error
            if (
                "CUDA error: an illegal memory access was encountered" in error_str
                or "cudaErrorIllegalAddress" in error_str
            ):

                from uuid import uuid4

                with open(f"/workspace/data/{str(uuid4())}.error", "w") as f:
                    for item in request.input:
                        f.write(f"{item.url}\n")

                print(f"FATAL CUDA error detected: {error_str}")
                print("Attempting to recover by reloading the model.")

                # Use the lock to ensure no other requests can interfere with the reload
                async with model_lock:
                    await load_model()

                # Inform the client that triggered the error to retry.
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,  # 503
                    detail="The model has been reloaded due to CUDA error. Please try your request again.",
                )
            else:
                # For all other errors, behave as before.
                print(f"An unexpected, non-CUDA error occurred during inference: {e}")
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail=f"An unexpected error occurred during inference: {str(e)}",
                )

    return OCRResponse(data=all_detections)
