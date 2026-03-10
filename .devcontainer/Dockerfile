# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.4.1-base-ubuntu22.04

FROM $BASE_IMG:$BASE_IMG_TAG AS base

RUN apt-get update && apt-get install -y \
      bzip2 \
      ca-certificates \
      curl \
      git \
      libgl1-mesa-glx \
      vim \
      wget \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/root/.local/bin:$PATH
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
    && uv venv --python 3.12 /opt/nv_ingest_runtime

ENV VIRTUAL_ENV=/opt/nv_ingest_runtime
ENV PATH=/opt/nv_ingest_runtime/bin:/root/.local/bin:$PATH

SHELL ["/bin/bash", "-c"]
