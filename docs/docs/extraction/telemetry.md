# Telemetry with NeMo Retriever Extraction

You can view telemetry data for [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## OpenTelemetry

After OpenTelemetry and Zipkin are running, you can open your browser to explore traces: 

- **Docker** — Use http://$YOUR_DOCKER_HOST:9411/zipkin/ 
- **Kubernetes** — Use http://$YOUR_K8S_OTEL_POD:9411/zipkin/

![](images/zipkin.png)

## Prometheus

After Prometheus is running, you can open your browser to explore metrics: 

- **Docker** — Use http://$YOUR_DOCKER_HOST:9090/ziplin/
- **Kubernetes** — Use http://$YOUR_K8S_OTEL_POD:9090/zipkin/

![](images/prometheus.png)
