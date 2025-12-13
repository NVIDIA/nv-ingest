# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git lfs install \
    && git clone https://huggingface.co/nvidia/nemotron-graphic-elements-v1 /opt/nemotron-graphic-elements-v1 \
    && pip install --no-cache-dir /opt/nemotron-graphic-elements-v1

COPY server.py .

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/server.py /app/server.py

WORKDIR /app

EXPOSE 8000 8001 8002

USER 65532:65532

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
