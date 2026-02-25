# Dockerfile for the Retriever online ingest Ray Serve API.
#
# IMPORTANT: Build from the nv-ingest repository ROOT (parent of retriever/), not from
# inside retriever/. The build context must include api/, client/, src/, and retriever/.
#
#   cd /path/to/nv-ingest
#   docker build -f retriever/Dockerfile -t retriever-online:latest .
#
# Uses NVIDIA CUDA 13 runtime base; uv installs and manages Python 3.12.

# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install uv (standalone) and build/runtime deps; no system Python 3.12 or deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="/root/.local/bin:$PATH" \
    && uv python install 3.12

ENV PATH="/root/.local/bin:$PATH"

# Copy full monorepo (context must be repo root so we get api/, client/, src/, retriever/)
COPY . /app/
RUN test -f /app/retriever/pyproject.toml && test -d /app/api && test -d /app/client && test -d /app/src || (echo "ERROR: Build context must be the nv-ingest repository root. Run: cd /path/to/nv-ingest && docker build -f retriever/Dockerfile -t IMAGE ." && exit 1)

WORKDIR /app/retriever

# Install project and all dependencies using uv-managed Python 3.12
RUN uv sync --no-dev --no-install-project --python 3.12 \
    && uv sync --no-dev --python 3.12

# Runtime stage: same CUDA 13 base; copy uv-managed Python so the venv works
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy uv-managed Python 3.12 from builder (venv's pyvenv.cfg points to this path)
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python

# Copy virtualenv and app
COPY --from=builder /app/retriever/.venv /app/.venv
COPY --from=builder /app/retriever /app/retriever
COPY --from=builder /app/api /app/api
COPY --from=builder /app/client /app/client
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

# LanceDB and model paths (override at run time if needed)
ENV ONLINE_LANCEDB_URI=/data/lancedb
ENV NEMOTRON_OCR_MODEL_DIR=/workspace/models/nemotron-ocr-v1

# Optional: remote embedding NIM (set to avoid loading local embed model)
# ENV ONLINE_EMBED_ENDPOINT=http://embedding:8000/v1

WORKDIR /app/retriever

EXPOSE 7670

CMD ["python", "-m", "retriever.online", "serve", "--host", "0.0.0.0", "--port", "7670"]
