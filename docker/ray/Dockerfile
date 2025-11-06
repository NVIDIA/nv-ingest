# docker/ray/Dockerfile
# Purpose: Build a Ray base image (for head/worker) that mirrors the nv-ingest
# conda environment and includes the nv-ingest API and client libraries.
# This does NOT reuse the nv-ingest runtime image; it recreates the environment
# using the same template as the root Dockerfile so we have parity.

ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619
ARG TARGETPLATFORM

FROM ${BASE_IMG}:${BASE_IMG_TAG} AS base

ARG TARGETPLATFORM
SHELL ["/bin/bash", "-lc"]

# OS dependencies (aligned with root Dockerfile)
RUN apt-get update && apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      wget \
    && apt-get clean

# ffmpeg (aligned with root Dockerfile)
COPY ./docker/scripts/install_ffmpeg.sh /tmp/install_ffmpeg.sh
RUN chmod +x /tmp/install_ffmpeg.sh \
    && bash /tmp/install_ffmpeg.sh \
    && rm /tmp/install_ffmpeg.sh

# Miniforge/conda
RUN wget -O /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Determine conda subdir based on architecture
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      export CONDA_SUBDIR=linux-aarch64; \
    else \
      export CONDA_SUBDIR=linux-64; \
    fi

# Install mamba, conda-build, and conda-merge
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    conda install -y mamba conda-build==24.5.1 conda-merge -n base -c conda-forge

# Copy environment specs (aligned with root Dockerfile)
COPY conda/environments/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.base.yml
COPY conda/environments/nv_ingest_environment.linux_64.yml /workspace/nv_ingest_environment.linux_64.yml
COPY conda/environments/nv_ingest_environment.linux_aarch64.yml /workspace/nv_ingest_environment.linux_aarch64.yml

# Create nv_ingest_runtime environment
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      conda-merge /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_aarch64.yml > /workspace/nv_ingest_environment.yml; \
      rm /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_aarch64.yml; \
    else \
      conda-merge /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_64.yml > /workspace/nv_ingest_environment.yml; \
      rm /workspace/nv_ingest_environment.base.yml /workspace/nv_ingest_environment.linux_64.yml; \
    fi; \
    mamba env create -f /workspace/nv_ingest_environment.yml

# Default shell and environment wiring
SHELL ["/bin/bash", "-c"]
RUN echo "source activate nv_ingest_runtime" >> ~/.bashrc
ENV LD_LIBRARY_PATH=/opt/conda/envs/nv_ingest_runtime/lib:$LD_LIBRARY_PATH

# Workdir
WORKDIR /workspace

# Build and install nv-ingest API and client wheels (no service wheel)
COPY ci ci
COPY api api
COPY client client

ENV HAYSTACK_TELEMETRY_ENABLED=False

# Install build tool
RUN source activate nv_ingest_runtime \
    && pip install --no-cache-dir 'build>=1.2.2'

# Build wheels for api and client
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    chmod +x ./ci/scripts/build_pip_packages.sh \
    && source activate nv_ingest_runtime \
    && ./ci/scripts/build_pip_packages.sh --type dev --lib api \
    && ./ci/scripts/build_pip_packages.sh --type dev --lib client

# Install built wheels
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    source activate nv_ingest_runtime \
    && pip install ./api/dist/*.whl \
    && pip install ./client/dist/*.whl

# Default command; docker-compose overrides with ray start for head/worker
CMD ["bash", "-lc", "sleep infinity"]
