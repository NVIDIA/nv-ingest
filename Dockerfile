# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.2.2-base-ubuntu22.04

# Use NVIDIA CUDA as the base image
FROM $BASE_IMG:$BASE_IMG_TAG AS base

# Set environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"

ARG UVICORN_WORKERS="32"

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Set up Miniconda environment
ENV PATH="/opt/miniconda/bin:$PATH"
RUN conda init bash
RUN echo "source /opt/miniconda/bin/activate" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

# Python & Poetry version files
COPY .poetry-version .poetry-version
COPY .python-version .python-version

# Add the necessary conda channels for Nvidia and RapidsAI
RUN conda config --env --add channels nvidia \
    && conda config --env --add channels rapidsai \
    && conda config --env --add channels nvidia/label/dev \
    && conda config --env --add channels pytorch \
    && conda config --env --add channels conda-forge

# Create the nv-ingest conda environment
RUN conda create --name nv-ingest python=$(cat .python-version) mamba

# Install conda packages before installing poetry & pip packages
RUN source activate nv-ingest \
    && mamba install -y -c nvidia/label/dev morpheus-core

# Install Poetry
RUN source activate nv-ingest \
    && pip install poetry==$(cat .poetry-version) \
    && poetry config virtualenvs.create false \
    && poetry --version

COPY pyproject.toml poetry.lock ./

# Prevent haystack from ending telemetry data
ENV HAYSTACK_TELEMETRY_ENABLED=False

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
ENV NV_INGEST_VERSION_OVERRIDE=${NV_INGEST_VERSION_OVERRIDE}
ENV NV_INGEST_CLIENT_VERSION_OVERRIDE=${NV_INGEST_VERSION_OVERRIDE}

# Cache the dependencies and install them before uploading source code changes
RUN source activate nv-ingest \
    && poetry install --with dev

# Copy the rest of the project files not omitted by the .dockerignore file
COPY . .

# Build the nv-ingest client
RUN source activate nv-ingest \
    && poetry build

FROM base AS runtime

# Install the client and library built by Poetry into the conda environment
RUN source activate nv-ingest \
    && pip install ./client/dist/*.whl \
    && rm -rf client/dist \
    && pip install ./dist/*.whl \
    && rm -rf ./dist

COPY ./docker/scripts/entrypoint_source_ext.sh /opt/docker/bin/entrypoint_source

# Start both the core nv-ingest pipeline service and teh FastAPI microservice in parallel
CMD ["sh", "-c", "python /workspace/pipeline.py & uvicorn nv_ingest.main:app --workers ${UVICORN_WORKERS} --host 0.0.0.0 --port 7670 & wait"]

FROM base AS development

RUN source activate nv-ingest && \
    pip install -e ./client

CMD ["/bin/bash"]
