# Notes:

retriever pipeline-to-csv --runner python --page-elements-http-endpoint http
://localhost:8000/v1/infer --graphic-elements-http-endpoint http://localhost:8003 --table-structure-http-endpoint http://localhost:8006 --ocr-http-endpoint http://localhost:8009/v1/infer --embedding-endpoint http://localhost:8012/v1 /datasets/nv-ingest/bo20 /raid/jdye
r/datasets/retriever-results/pipeline-to-csv-bo20-results.csv



## Running Stages:

1. retriever pdf stage page-elements --config retriever/pdf_stage_config.yaml
2. 
