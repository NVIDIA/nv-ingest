# Dockerfile for the Retriever online ingest Ray Serve API.
# Build from the repository root so path dependencies (api, client, src) resolve:
#   docker build -f retriever/Dockerfile .
#
# Uses NVIDIA CUDA 13 runtime base with Python 3.12 for GPU-accelerated pipeline
# (page elements, OCR, embedding). To use a CPU-only build, override BASE_IMAGE:
#   docker build -f retriever/Dockerfile --build-arg BASE_IMAGE=python:3.12-slim-bookworm .

# syntax=docker/dockerfile:1.4
# CUDA 13 + Ubuntu 22.04; Python 3.12 is installed in the stages below
ARG BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install Python 3.12 (deadsnakes PPA) and build/runtime deps for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip and uv for Python 3.12; set python3 to 3.12 so uv uses it
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install --no-cache-dir uv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy monorepo layout so retriever's path deps (../api, ../client, ../src) resolve
COPY api /app/api
COPY client /app/client
COPY src /app/src
COPY retriever /app/retriever

WORKDIR /app/retriever

# Install project and all dependencies (uses [tool.uv.sources] for path deps and extra indexes)
RUN uv sync --no-dev --no-install-project \
    && uv sync --no-dev

# Runtime stage: same CUDA 13 base, Python 3.12, minimal runtime deps
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy virtualenv from builder (uv places it in .venv by default)
COPY --from=builder /app/retriever/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy installed packages' code (retriever and path deps are in the venv)
COPY --from=builder /app/retriever /app/retriever
COPY --from=builder /app/api /app/api
COPY --from=builder /app/client /app/client
COPY --from=builder /app/src /app/src

# LanceDB and model paths (override at run time if needed)
ENV ONLINE_LANCEDB_URI=/data/lancedb
# Mount the Nemotron OCR model dir or set NEMOTRON_OCR_MODEL_DIR to a path that exists in the container
ENV NEMOTRON_OCR_MODEL_DIR=/workspace/models/nemotron-ocr-v1

# Optional: remote embedding NIM (set to avoid loading local embed model)
# ENV ONLINE_EMBED_ENDPOINT=http://embedding:8000/v1

WORKDIR /app/retriever

EXPOSE 7670

# Start the online ingest Ray Serve API; bind all interfaces for container networking
CMD ["python", "-m", "retriever.online", "serve", "--host", "0.0.0.0", "--port", "7670"]
