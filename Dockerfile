# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.2.2-base-ubuntu22.04

# Use NVIDIA Morpheus as the base image
FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"

# Set the working directory in the container
WORKDIR /workspace

# Copy custom entrypoint script
COPY ./docker/scripts/entrypoint.sh /workspace/docker/entrypoint.sh

# .dockerignore will catch any files that we do not want to include in
# image. This prevents the need for multiple rm statements in the Dockerfile
COPY . .

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      wget \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
    && apt-get clean

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Install Mamba, a faster alternative to conda, within the base environment
RUN conda install -y mamba -n base -c conda-forge

# Create nv_ingest base environment
RUN mamba create -y --name nv_ingest \
    python=$(cat .python-version) \
    poetry=$(cat .poetry-version) \
    tini

# Activate the environment (make it default for subsequent commands)
RUN echo "source activate nv_ingest" >> ~/.bashrc

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Install Morpheus dependencies
RUN source activate nv_ingest \
    && mamba install -y \
     nvidia/label/dev::morpheus-core \
     nvidia/label/dev::morpheus-llm \
     # pin to earlier version of cuda-python until __pyx_capi__ fix is upstreamed.
     cuda-python=12.6.0 \
     -c rapidsai -c pytorch -c nvidia -c conda-forge

FROM base AS nv_ingest_install

# Prevent haystack from sending telemetry data
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

SHELL ["/bin/bash", "-c"]

# Cache the requirements and install them before uploading source code changes
RUN source activate nv_ingest \
    && poetry install --with runtime,client,dev,extra \
    && poetry build

# Install patched MRC version to circumvent NUMA node issue -- remove after Morpheus 10.24 release
RUN source activate nv_ingest \
    && conda install -y -c nvidia/label/dev mrc=24.10.00a=cuda_12.5_py310_h5ae46af_10

FROM nv_ingest_install AS runtime

COPY src/pipeline.py ./

RUN chmod +x /workspace/docker/entrypoint.sh

# Set entrypoint to tini with a custom entrypoint script
ENTRYPOINT ["/opt/conda/envs/nv_ingest/bin/tini", "--", "/workspace/docker/entrypoint.sh"]

# Start both the core nv-ingest pipeline service and the FastAPI microservice in parallel
CMD ["sh", "-c", "python /workspace/pipeline.py & uvicorn nv_ingest.main:app --workers 32 --host 0.0.0.0 --port 7670 & wait"]

FROM nv_ingest_install AS development

RUN source activate nv_ingest && \
    pip install -e ./client

CMD ["/bin/bash"]
