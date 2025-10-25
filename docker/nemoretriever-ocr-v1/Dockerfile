# Start from the official NVIDIA PyTorch container to ensure CUDA compatibility
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install Git and Git LFS
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# This allows the token to be passed securely during the build process.
# It will NOT be available in the final running container.
ARG HF_TOKEN

# Set the working directory
WORKDIR /app

# Clone the model repository from Hugging Face
RUN git lfs install && \
    git clone https://huggingface.co/nvidia/nemoretriever-ocr-v1

# Set the working directory to the cloned repository
WORKDIR /app/nemoretriever-ocr-v1/nemo-retriever-ocr

## Install the model's custom dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip hatchling "setuptools>=68" --root-user-action ignore

# Ensure no prebuilt binaries/artifacts from the host are present
RUN rm -f src/nemo_retriever_ocr_cpp/*.so || true \
    && rm -rf build/ dist/

RUN --mount=type=cache,target=/root/.cache/pip \
    BUILD_CPP_FORCE=1 pip install -v . --no-build-isolation --root-user-action ignore

WORKDIR /app/nemoretriever-ocr-v1

# Copy our new server script into the container
COPY server.py .

# Install dependencies for our advanced FastAPI server
# Pillow is needed for image processing from bytes
RUN pip install fastapi uvicorn pydantic python-multipart pillow

# Expose the port the server will listen on
EXPOSE 8000

# Define the command to start the Uvicorn server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
