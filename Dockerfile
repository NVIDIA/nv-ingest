# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/morpheus/morpheus
ARG BASE_IMG_TAG=v24.06.01-runtime

# Use NVIDIA Morpheus as the base image
FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"

# Set the working directory in the container
WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes \
    libgl1-mesa-glx

# Copy the module code
COPY setup.py setup.py
# Don't copy full source here, pipelines won't be installed via setup anyway, and this allows us to rebuild more quickly if we're just changing the pipeline
COPY src/nv_ingest src/nv_ingest
COPY client client
COPY ci ci
COPY requirements.txt test-requirements.txt util-requirements.txt ./

RUN rm -rf ./src/nv_ingest/dist ./client/dist

SHELL ["/bin/bash", "-c"]

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

# Run the build_pip_packages.sh script with the specified build type and library
RUN chmod +x ./ci/scripts/build_pip_packages.sh \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib client \
    && ./ci/scripts/build_pip_packages.sh --type ${RELEASE_TYPE} --lib service

RUN source activate morpheus \
    && pip install ./dist/*.whl

RUN source activate morpheus \
    && rm -rf src requirements.txt test-requirements.txt util-requirements.txt

# Interim pyarrow backport until folded into upstream dependency tree
RUN source activate morpheus \
    && conda install https://anaconda.org/conda-forge/pyarrow/14.0.2/download/linux-64/pyarrow-14.0.2-py310h188ebfb_19_cuda.conda

# Upgrade setuptools to mitigate https://github.com/advisories/GHSA-cx63-2mw6-8hw5
RUN source activate base \
    && conda install setuptools==70.0.0

FROM base AS runtime

RUN source activate morpheus \
    && pip install ./client/dist/*.whl \
    && rm -rf client/dist

COPY src/pipeline.py ./
COPY pyproject.toml ./
COPY ./docker/scripts/entrypoint_source_ext.sh /opt/docker/bin/entrypoint_source

CMD ["python", "/workspace/pipeline.py"]

FROM base AS development

RUN source activate morpheus && \
    pip install -e ./client

CMD ["/bin/bash"]
