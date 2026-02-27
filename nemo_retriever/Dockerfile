# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# syntax=docker/dockerfile:1.3
#
# Build from nv-ingest root: docker build -f nemo_retriever/Dockerfile -t retriever .
# Run: docker run retriever  (help) or docker run -v /host/docs:/data retriever /data

ARG BASE_IMG=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMG_TAG=jammy-20250619

FROM $BASE_IMG:$BASE_IMG_TAG AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      ca-certificates \
      curl \
      libgl1-mesa-glx \
      libglib2.0-0 \
      wget \
    && apt-get clean

# LibreOffice (headless) for docx/pptx -> PDF. GPL source handling per nv-ingest Dockerfile.
ARG GPL_LIBS="\
    libltdl7 \
    libhunspell-1.7-0 \
    libhyphen0 \
    libdbus-1-3 \
"
# Keep libfreetype6 so LibreOffice (soffice.bin) can load; omit from force-remove.
ARG FORCE_REMOVE_PKGS="\
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

# Miniforge + conda env with Python 3.12, CUDA 13 cudart, uv (for OCR: libcudart.so.13)
RUN wget -q -O /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

COPY conda/environments/retriever_libcudart.yml /tmp/retriever_libcudart.yml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda env create -f /tmp/retriever_libcudart.yml \
    && rm /tmp/retriever_libcudart.yml

# So Nemotron OCR finds libcudart.so.13 at runtime
ENV LD_LIBRARY_PATH=/opt/conda/envs/retriever_libcudart/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["/bin/bash"]
