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
ARG GIT_COMMIT
LABEL git_commit=$GIT_COMMIT

RUN apt-get update && apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      make \
      tini \
      wget \
    && apt-get clean

COPY ./docker/scripts/install_ffmpeg.sh scripts/install_ffmpeg.sh
RUN chmod +x scripts/install_ffmpeg.sh \
    && bash scripts/install_ffmpeg.sh \
    && rm scripts/install_ffmpeg.sh

# Install libreoffice
# For GPL-licensed components, we provide their source code in the container
# via `apt-get source` below to satisfy GPL requirements.
ARG GPL_LIBS="\
    libltdl7 \
    libhunspell-1.7-0 \
    libhyphen0 \
    libdbus-1-3 \
"
ARG FORCE_REMOVE_PKGS="\
    libfreetype6 \
    ucf \
    liblangtag-common \
    libjbig0 \
    pinentry-curses \
    gpg-agent \
    gnupg-utils \
    gpgsm \
    gpg-wks-server \
    gpg-wks-client \
    gpgconf \
    gnupg \
    readline-common \
    libreadline8 \
    dirmngr \
    libjpeg8 \
"
RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      dpkg-dev \
      libreoffice \
      $GPL_LIBS \
    && apt-get source $GPL_LIBS \
    && for pkg in $FORCE_REMOVE_PKGS; do \
         dpkg --remove --force-depends "$pkg" || true; \
       done \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/root/.local/bin:$PATH
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
    && uv venv --python 3.12 /opt/nv_ingest_runtime

ENV VIRTUAL_ENV=/opt/nv_ingest_runtime
ENV PATH=/opt/nv_ingest_runtime/bin:/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/nv_ingest_runtime/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

FROM base AS nv_ingest_install

COPY ci ci

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

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install 'build>=1.2.2'

RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x ./ci/scripts/build_pip_packages.sh \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib api \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install ./src/dist/*.whl \
    && uv pip install ./api/dist/*.whl \
    && uv pip install ./client/dist/*.whl

RUN rm -rf src

FROM nv_ingest_install AS runtime

COPY src/microservice_entrypoint.py ./
COPY config/default_pipeline.yaml ./config/

COPY ./docker/scripts/entrypoint.sh /workspace/docker/entrypoint.sh
COPY ./docker/scripts/entrypoint_source_ext.sh /workspace/docker/entrypoint_source_ext.sh
COPY ./docker/scripts/post_build_triggers.py /workspace/docker/post_build_triggers.py

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=hf_token,required=false \
    python3 /workspace/docker/post_build_triggers.py

RUN chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/docker/entrypoint.sh"]

FROM runtime AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    WHEEL="$(ls ./api/dist/*.whl)" \
    && uv pip install "${WHEEL}[test]"

FROM nv_ingest_install AS development

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ./client

CMD ["/bin/bash"]

FROM nv_ingest_install AS docs

COPY docs docs

# Docs needs all the source code present so add it to the container
COPY src src
COPY api api
COPY client client

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r ./docs/requirements.txt

CMD ["bash", "-c", "cd /workspace/docs && make docs"]
