# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=ubuntu
ARG BASE_IMG_TAG=22.04

FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"
ARG DOWNLOAD_LLAMA_TOKENIZER=""
ARG HF_ACCESS_TOKEN=""
ARG MODEL_PREDOWNLOAD_PATH=""

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Embed the `git rev-parse HEAD` as a Docker metadata label
# Allows for linking container builds to git commits
# docker inspect nv-ingest:latest | jq '.[0].Config.Labels.git_commit' -> GIT_SHA
ARG GIT_COMMIT
LABEL git_commit=$GIT_COMMIT

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    software-properties-common \
    python3.12 python3.12-venv python3.12-dev \
    bzip2 \
    ca-certificates \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip manually to get correct version
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --config python
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1

# Set the working directory in the container
WORKDIR /workspace

FROM base AS nv_ingest_install
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
ENV NV_INGEST_VERSION_OVERRIDE=${NV_INGEST_VERSION_OVERRIDE}

SHELL ["/bin/bash", "-c"]

COPY tests tests
COPY data data
COPY api api
COPY client client
COPY src/nv_ingest src/nv_ingest
RUN rm -rf ./src/nv_ingest/dist ./client/dist ./api/dist

# Install Python build utility for building wheels
RUN pip install 'build>=1.2.2'


RUN chmod +x ./ci/scripts/build_pip_packages.sh \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib api \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN pip install ./dist/*.whl \
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

RUN python /workspace/docker/post_build_triggers.py

RUN chmod +x /workspace/docker/entrypoint.sh

# # Set entrypoint to tini with a custom entrypoint script
# ENTRYPOINT ["/opt/conda/envs/nv_ingest_runtime/bin/tini", "--", "/workspace/docker/entrypoint.sh"]

FROM nv_ingest_install AS development

RUN pip install -e ./client

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

RUN pip install -r ./docs/requirements.txt

# Default command: Run `make docs`
CMD ["bash", "-c", "cd /workspace/docs && source activate nv_ingest_runtime && make docs"]
