<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Telemetry

## Docker compose

To run OpenTelemetry locally, run

```shell
$ docker compose up otel-collector
```

Once and OpenTelemetry and Zipkin are running, you can open your browser to explore traces: http://$YOUR_DOCKER_HOST:9411/zipkin/.

![](images/zipkin.png)

To run Prometheus, run

```shell
$ docker compose up prometheus
```

Once Promethus is running, you can open your browser to explore metrics: [http://$YOUR_DOCKER_HOST:9090/]

![](images/prometheus.png)
