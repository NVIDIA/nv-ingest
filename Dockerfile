# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619
ARG TARGETPLATFORM

FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG TARGETPLATFORM
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

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      wget \
    && apt-get clean

# Install ffmpeg
COPY ./docker/scripts/install_ffmpeg.sh scripts/install_ffmpeg.sh
RUN chmod +x scripts/install_ffmpeg.sh \
    && bash scripts/install_ffmpeg.sh \
    && rm scripts/install_ffmpeg.sh

# Install micromamba, a faster alternative to conda at /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64) m_arch="64" ;; \
      aarch64) m_arch="aarch64" ;; \
      *) echo "Unsupported arch: $arch" && exit 1 ;; \
    esac; \
    curl -L "https://micro.mamba.pm/api/micromamba/linux-${m_arch}/latest" -o /tmp/micromamba.tar.bz2; \
    mkdir -p /usr/local/bin; \
    tar -xvjf /tmp/micromamba.tar.bz2 -C /usr/local/bin --strip-components=1 bin/micromamba; \
    rm -f /tmp/micromamba.tar.bz2; \
    mkdir -p "$MAMBA_ROOT_PREFIX"

# Cache mounts:
#   - /opt/conda/pkgs : conda/mamba/micromamba package cache
#   - /root/.cache/pip: pip cache
# We’ll use micromamba-run instead of "source activate".
SHELL ["/bin/bash", "-lc"]

COPY conda/environments/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.base.yml
COPY conda/environments/nv_ingest_environment.linux_64.yml /workspace/nv_ingest_environment.linux_64.yml
COPY conda/environments/nv_ingest_environment.linux_aarch64.yml /workspace/nv_ingest_environment.linux_aarch64.yml

# Set `extract_threads 1` for QEMU+ARM build
# https://github.com/mamba-org/mamba/issues/1611
RUN micromamba config set extract_threads 1

# Install conda-merge into base so we can merge YAMLs
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n base -c conda-forge conda-merge

# Merge env files per-arch and create nv_ingest base environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      micromamba run -n base conda-merge /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_aarch64.yml > /workspace/nv_ingest_environment.yml; \
      rm /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_aarch64.yml; \
    else \
      micromamba run -n base conda-merge /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_64.yml > /workspace/nv_ingest_environment.yml; \
      rm /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_64.yml; \
    fi; \
    micromamba create -y -n nv_ingest_runtime -f /workspace/nv_ingest_environment.yml

# Install tini in the runtime env
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n nv_ingest_runtime -c conda-forge tini

# Ensure dynamically linked libraries in the conda environment are found at runtime
ENV PATH=/opt/conda/envs/nv_ingest_runtime/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/nv_ingest_runtime/lib:$LD_LIBRARY_PATH

# Set the working directory in the container
WORKDIR /workspace

FROM base AS nv_ingest_install
# Copy the module code
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

COPY scripts scripts
COPY tests tests
COPY data data
COPY api api
COPY client client
COPY src src
RUN rm -rf ./src/nv_ingest/dist ./src/dist ./client/dist ./api/dist

# Install python build from pip, version needed not present in conda
RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n nv_ingest_runtime pip install 'build>=1.2.2'

# Add pip cache path to match conda's package cache
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    chmod +x ./ci/scripts/build_pip_packages.sh \
 && micromamba run -n nv_ingest_runtime ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib api \
 && micromamba run -n nv_ingest_runtime ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
 && micromamba run -n nv_ingest_runtime ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n nv_ingest_runtime pip install ./src/dist/*.whl \
 && micromamba run -n nv_ingest_runtime pip install ./api/dist/*.whl \
 && micromamba run -n nv_ingest_runtime pip install ./client/dist/*.whl

RUN rm -rf src

FROM nv_ingest_install AS runtime

COPY src/microservice_entrypoint.py ./
COPY config/default_pipeline.yaml ./config/

# Copy entrypoint script(s)
COPY ./docker/scripts/entrypoint.sh /workspace/docker/entrypoint.sh
COPY ./docker/scripts/entrypoint_source_ext.sh /workspace/docker/entrypoint_source_ext.sh

# Copy post build triggers script
COPY ./docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

RUN  --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n nv_ingest_runtime python3 /workspace/docker/post_build_triggers.py

# Remove graphviz and its dependencies to reduce image size
RUN micromamba remove -y -n nv_ingest_runtime graphviz python-graphviz || true && \
    micromamba remove -y -n nv_ingest_runtime gtk3 pango cairo fonts-conda-ecosystem || true

RUN chmod +x /workspace/docker/entrypoint.sh

# Set entrypoint to tini with a custom entrypoint script
ENTRYPOINT ["/opt/conda/envs/nv_ingest_runtime/bin/tini", "--", "/workspace/docker/entrypoint.sh"]

FROM nv_ingest_install AS development

RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n nv_ingest_runtime pip install -e ./client

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

RUN --mount=type=cache,target=/root/.cache/pip \
    micromamba run -n nv_ingest_runtime pip install -r ./docs/requirements.txt

# Default command: Run `make docs`
CMD ["bash", "-c", "cd /workspace/docs && source activate nv_ingest_runtime && make docs"]
