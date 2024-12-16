# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3

ARG BASE_IMG=nvcr.io/nvidia/cuda
ARG BASE_IMG_TAG=12.4.1-base-ubuntu22.04

# Use NVIDIA cuda
FROM $BASE_IMG:$BASE_IMG_TAG AS base

ARG RELEASE_TYPE="dev"
ARG VERSION=""
ARG VERSION_REV="0"

# Install necessary dependencies using apt-get
RUN apt-get update && apt-get install -y \
      wget \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      vim \
      git \
    && apt-get clean

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Install Mamba, a faster alternative to conda, within the base environment
RUN conda install -y mamba -n base -c conda-forge

COPY conda/environments/nv_ingest_environment.yml /workspace/nv_ingest_environment.yml

# Create nv_ingest base environment
RUN mamba env create -f /workspace/nv_ingest_environment.yml \
    && conda clean --all --yes

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Activate the environment (make it default for subsequent commands)
RUN echo "source activate nv_ingest_runtime" >> ~/.bashrc
