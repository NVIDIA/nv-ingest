# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.5.1-base-ubuntu22.04

# Use NVIDIA Morpheus as the base image
FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"
ARG DOWNLOAD_LLAMA_TOKENIZER=""
ARG HF_ACCESS_TOKEN=""
ARG MODEL_PREDOWNLOAD_PATH=""

# Embed the `git rev-parse HEAD` as a Docker metadata label
# Allows for linking container builds to git commits
# docker inspect nv-ingest:latest | jq '.[0].Config.Labels.git_commit' -> GIT_SHA
ARG GIT_COMMIT
LABEL git_commit=$GIT_COMMIT

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      wget \
    && apt-get clean

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Install Mamba, a faster alternative to conda, within the base environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    conda install -y mamba conda-build==24.5.1 -n base -c conda-forge

COPY conda/environments/nv_ingest_environment.yml /workspace/nv_ingest_environment.yml
# Create nv_ingest base environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    mamba env create -f /workspace/nv_ingest_environment.yml

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate the environment (make it default for subsequent commands)
RUN echo "source activate nv_ingest_runtime" >> ~/.bashrc

# Install Tini via conda from the conda-forge channel
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    source activate nv_ingest_runtime \
    && mamba install -y -c conda-forge tini

# Ensure dynamically linked libraries in the conda environment are found at runtime
ENV LD_LIBRARY_PATH=/opt/conda/envs/nv_ingest_runtime/lib:$LD_LIBRARY_PATH

# Set the working directory in the container
WORKDIR /workspace

FROM base AS nv_ingest_install
# Copy the module code
COPY setup.py setup.py
COPY ci ci

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

SHELL ["/bin/bash", "-c"]

COPY tests tests
COPY data data
COPY api api
COPY client client
COPY src/nv_ingest src/nv_ingest
RUN rm -rf ./src/nv_ingest/dist ./client/dist ./api/dist

# Install python build from pip, version needed not present in conda
RUN source activate nv_ingest_runtime \
    && pip install 'build>=1.2.2'

# Add pip cache path to match conda's package cache
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    chmod +x ./ci/scripts/build_pip_packages.sh \
    && source activate nv_ingest_runtime \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib api \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN --mount=type=cache,target=/opt/conda/pkgs\
    --mount=type=cache,target=/root/.cache/pip \
    source activate nv_ingest_runtime \
    && pip install ./dist/*.whl \
    && pip install ./api/dist/*.whl \
    && pip install ./client/dist/*.whl


RUN rm -rf src

FROM nv_ingest_install AS runtime

COPY src/microservice_entrypoint.py ./
COPY pyproject.toml ./

# Copy entrypoint script(s)
COPY ./docker/scripts/entrypoint.sh /workspace/docker/entrypoint.sh
COPY ./docker/scripts/entrypoint_source_ext.sh /workspace/docker/entrypoint_source_ext.sh

# Copy post build triggers script
COPY ./docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

RUN  --mount=type=cache,target=/root/.cache/pip \
    source activate nv_ingest_runtime \
    && python3 /workspace/docker/post_build_triggers.py

# Remove graphviz and its dependencies to reduce image size
RUN source activate nv_ingest_runtime && \
    mamba remove graphviz python-graphviz --force -y && \
    mamba uninstall gtk3 pango cairo fonts-conda-ecosystem -y

RUN chmod +x /workspace/docker/entrypoint.sh

# Set entrypoint to tini with a custom entrypoint script
ENTRYPOINT ["/opt/conda/envs/nv_ingest_runtime/bin/tini", "--", "/workspace/docker/entrypoint.sh"]

FROM nv_ingest_install AS development

RUN source activate nv_ingest_runtime && \
    --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    pip install -e ./client

CMD ["/bin/bash"]


FROM nv_ingest_install AS docs

# Install dependencies needed for docs generation
RUN apt-get update && apt-get install -y \
      make \
    && apt-get clean

COPY docs docs

# Docs needs all the source code present so add it to the container
COPY src src
COPY api api
COPY client client

RUN source activate nv_ingest_runtime && \
    pip install -r ./docs/requirements.txt

# Default command: Run `make docs`
CMD ["bash", "-c", "cd /workspace/docs && source activate nv_ingest_runtime && make docs"]
