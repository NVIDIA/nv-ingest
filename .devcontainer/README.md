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

# Morpheus Devcontainer

The Morpheus devcontainer is provided as a quick-to-set-up development and exploration environment for use with [Visual Studio Code](https://code.visualstudio.com) (Code). The devcontainer is a lightweight container which mounts-in a Conda environment with cached packages, alleviating long Conda download times on subsequent launches. It provides a simple framework for adding developer-centric [scripts](#development-scripts), and incorporates some helpful Code plugins, such as clangd and CMake support.

More information about devcontainers can be found at [`containers.dev`](https://containers.dev/).

## Getting Started

To get started, simply open the Morpheus repository root folder within Code. A window should appear at the bottom-right corner of the editor asking if you would like to reopen the workspace inside of the dev container. After clicking the confirmation dialog, the container will first build, then launch, then remote-attach.

If the window does not appear, or you would like to rebuild the container, click ctrl-shift-p and search for `Dev Containers: Rebuild and Reopen in Container`. Hit enter, and the container will first build, then launch, then remote-attach.

Once connected to the devcontainer within code, the `setup-morpheus-env` script will begin to run and solve a Morpheus Conda environment (this Conda environment is local to the Morpheus repository and dev container and will not override any host environments). You should see the script executing in one of Code's integrated terminal. Once the script has completed, we're ready to start development or exploration of Morpheus. By default, each _new_ integrated terminal will automatically Conda activate the Morpheus environment.

## Development Scripts
Several convenient scripts are available in the devcontainer's `PATH` (`.devcontainer/bin`) for starting, stopping, and interacting with Triton and Kafka. More scripts can be added as needed.

### Interacting with Triton
To start Triton and connect it to the devcontainer network, the `dev-triton-start` script can be used. The following example starts _or restarts_ Triton with the `abp-pcap-xgb` model loaded.
```
dev-triton-start abp-pcap-xgb
```
Triton should now be started and DNS resolvable as `triton`.
```
ping triton
```
To load a different model, simply call `dev-triton-start` with a different model name. Multiple models can be loaded simultaneously by adding more model names to the command.
```
dev-triton-start model-1 model-2 ... model-n
```
To stop Triton, call `dev-triton-stop`. This may take several seconds as the Triton server shuts down gracefully.
```
dev-triton-stop
```
### Interacting with Kafka
To start Kafka and connect it to the devcontainer network, the `dev-kafka-start` script can be used. The following example starts _or restarts_ Kafka and Zookeeper.
```
dev-kafka-start
```
Kafka should now be started and DNS resolvable as `kafka`.
```
ping kafka
```
It can be extremely useful to interact directly with Kafka to produce test data to a specific topic. To do this, the `dev-kafka-produce` script can be used. This script opens a connect to the Kafka server and starts `kafka-console-producer.sh`. In this case with a specific topic. Once started, a prompt will appear in which new-line delimited messages can be entered in to the console.
```
dev-kafka-produce test-topic
>
```
It can also be useful to produce messages from a file. For this, the `dev-kafka-produce` script can be used with two arguments. The first argument is the topic name, and the second argument is the file to be forwarded to the console producer.
```
dev-kafka-produce test-topic $MORPHEUS_ROOT/examples/data/pcap_dump.jsonlines
```
To retrieve the logs from Kafka and Zookeeper, use the `dev-kafka-logs` script.
```
dev-kafka-logs
```
To stop Kafka and Zookeeper, use the `dev-kafka-stop` script.
```
dev-kafka-stop
```
