# Telemetry with NV-Ingest

## OpenTelemetry

After OpenTelemetry and Zipkin are running, you can open your browser to explore traces: 

- For deployment via docker: http://$YOUR_DOCKER_HOST:9411/zipkin/ 
- For deployment via k8s: http://$YOUR_K8S_OTEL_POD:9411/zipkin/

![](images/zipkin.png)

## Prometheus

After Prometheus is running, you can open your browser to explore metrics: 

- For deployment via docker: http://$YOUR_DOCKER_HOST:9090/ziplin/
- For deployment via k8s: http://$YOUR_K8S_OTEL_POD:9090/zipkin/

![](images/prometheus.png)
