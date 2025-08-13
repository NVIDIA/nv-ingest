# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"
ARG DOWNLOAD_LLAMA_TOKENIZER="False"
ARG HF_ACCESS_TOKEN=""
ARG MODEL_PREDOWNLOAD_PATH="/workspace/models/"

# Embed the `git rev-parse HEAD` as a Docker metadata label
# Allows for linking container builds to git commits
# docker inspect nv-ingest:latest | jq '.[0].Config.Labels.git_commit' -> GIT_SHA
ARG GIT_COMMIT
LABEL git_commit=$GIT_COMMIT

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies using apt-get
RUN apt-get update && \
    apt-get install -y \
      software-properties-common \
      curl \
      gnupg && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      wget \
      tini \
      python3.12 \
      python3.12-venv \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default `python3` (optional)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set the working directory in the container
WORKDIR /workspace

FROM base AS nv_ingest_install

# Create a virtual environment
RUN python3 -m venv /opt/venv/nv-ingest

ENV PATH="/opt/venv/nv-ingest/bin:$PATH"

# Copy the module code
COPY ci ci

# Ensure the NV_INGEST_VERSION is PEP 440 compatible
RUN if [ -z "${VERSION}" ]; then \
        export VERSION="$(date +'%Y.%m.%d')"; \
    fi; \
    if [ "${RELEASE_TYPE}" = "dev" ]; then \
        export NV_INGEST_VERSION_OVERRIDE="${VERSION}.dev${VERSION_REV}"; \
    elif [ "${RELEASE_TYPE}" = "release" ]; then \
        export NV_INGEST_VERSION_OVERRIDE="${VERSION}.post${VERSION_REV}"; \
    else \
        echo "Invalid RELEASE_TYPE: ${RELEASE_TYPE}"; \
        exit 1; \
    fi

ENV NV_INGEST_RELEASE_TYPE=${RELEASE_TYPE}

SHELL ["/bin/bash", "-c"]

COPY tests tests
COPY data data
RUN rm -rf ./data/dev # Ensure dev directory is not copied
COPY api api
COPY client client
COPY src src
RUN rm -rf ./src/nv_ingest/dist ./src/dist ./client/dist ./api/dist

# Install Pip dependencies needed for the build
RUN pip install "build>=1.2.2"

# Add pip cache path to match conda's package cache
RUN --mount=type=cache,target=/root/.cache/pip \
    chmod +x ./ci/scripts/build_pip_packages.sh \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib api \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ./src/dist/*.whl \
    && pip install ./api/dist/*.whl \
    && pip install ./client/dist/*.whl


RUN rm -rf src

FROM nv_ingest_install AS rest-api

RUN pip install python-multipart

# Copy entrypoint script(s)
COPY ./docker/scripts/entrypoint-rest-api.sh /workspace/docker/entrypoint-rest-api.sh
COPY ./docker/scripts/entrypoint_source_ext.sh /workspace/docker/entrypoint_source_ext.sh

# Copy post build triggers script
COPY ./docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

RUN  --mount=type=cache,target=/root/.cache/pip \
    python3 /workspace/docker/post_build_triggers.py

RUN chmod +x /workspace/docker/entrypoint-rest-api.sh

# Set entrypoint to tini with a custom entrypoint script
ENTRYPOINT ["tini", "--", "/workspace/docker/entrypoint-rest-api.sh"]


FROM nv_ingest_install AS worker

# Copy entrypoint script(s)
COPY ./docker/scripts/entrypoint-worker.sh /workspace/docker/entrypoint-worker.sh
COPY ./docker/scripts/entrypoint_source_ext.sh /workspace/docker/entrypoint_source_ext.sh

# Copy post build triggers script
COPY ./docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

RUN  --mount=type=cache,target=/root/.cache/pip \
    python3 /workspace/docker/post_build_triggers.py

RUN chmod +x /workspace/docker/entrypoint-worker.sh

# Set entrypoint to tini with a custom entrypoint script
ENTRYPOINT ["tini", "--", "/workspace/docker/entrypoint-worker.sh"]
