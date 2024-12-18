<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# NV-Ingest Devcontainer

The nv-ingest devcontainer is provided as a quick-to-set-up development and exploration environment for use with [Visual Studio Code](https://code.visualstudio.com) (Code). The devcontainer is a lightweight container which mounts-in a Conda environment with cached packages, alleviating long Conda download times on subsequent launches. It provides a simple framework for adding developer-centric [scripts](#development-scripts), and incorporates some helpful Code plugins.

More information about devcontainers can be found at [`containers.dev`](https://containers.dev/).

## Getting Started

To get started, simply open the nv-ingest repository root folder within Code. A window should appear at the bottom-right corner of the editor asking if you would like to reopen the workspace inside of the dev container. After clicking the confirmation dialog, the container will first build, then launch, then remote-attach.

If the window does not appear, or you would like to rebuild the container, click ctrl-shift-p and search for `Dev Containers: Rebuild and Reopen in Container`. Hit enter, and the container will first build, then launch, then remote-attach.
