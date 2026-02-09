# Notes:

retriever pipeline-to-csv --runner python --page-elements-http-endpoint http
://localhost:8000/v1/infer --graphic-elements-http-endpoint http://localhost:8003 --table-structure-http-endpoint http://localhost:8006 --ocr-http-endpoint http://localhost:8009/v1/infer --embedding-endpoint http://localhost:8012/v1 /datasets/nv-ingest/bo20 /raid/jdye
r/datasets/retriever-results/pipeline-to-csv-bo20-results.csv



## Running Stages:

retriever local stage3 run --input /home/local/jdyer/datasets/jp20-results-hf-standalone/

1. retriever local stage1 page-elements --config retriever/pdf_stage_config.yaml
2. retriever local stage2 run --config ./retriever/infographic_stage_config.yaml --input /home/local/jdyer/datasets/jp20-results-hf-standalone/
3. retriever local stage3 run --input /home/local/jdyer/datasets/jp20-results-hf-standalone/
4. retriever local stage4 run --input /home/local/jdyer/datasets/jp20-results-hf-standalone/
5. retriever local stage5 run --input-dir /home/local/jdyer/datasets/jp20-results-hf-standalone/ --endpoint-url http://localhost:8012/v1
6. retriever local stage6 run --input-dir /home/local/jdyer/datasets/jp20-results-hf-standalone/
7. retriever local stage7 run --query-csv jp20_query_gt.csv --embedding-endpoint http://localhost:8012/v1
7a. retriever local stage7 recall-with-main --query-csv bo767_query_gt.csv


retriever local stage1 page-elements --config retriever/pdf_stage_config.yaml && retriever local stage2 run --config ./retriever/infographic_stage_config.yaml --input /home/local/jdyer/datasets/bo767-results-hf-standalone/ && retriever local stage3 run --input /home/local/jdyer/datasets/bo767-results-hf-standalone/ && retriever local stage4 run --input /home/local/jdyer/datasets/bo767-results-hf-standalone/ && retriever local stage5 run --input-dir /home/local/jdyer/datasets/bo767-results-hf-standalone/ --endpoint-url http://localhost:8012/v1 && retriever local stage6 run --input-dir /home/local/jdyer/datasets/bo767-results-hf-standalone/ && retriever local stage7 run --query-csv bo767_query_gt.csv --embedding-endpoint http://localhost:8012/v1

1-8-2026
Recall metrics
  recall@1: 0.2311
  recall@5: 0.4208
  recall@10: 0.4985


1-8-2026 recall-with-main logic
    Recall@1: 0.04742684157416751
    Recall@5: 0.24318869828456105
    Recall@10: 0.3350151362260343

# Ray Running

python ./retriever/src/retriever/ingest-batch-pipeline.py --input-dir /home/local/jdyer/datasets/bo767 --method pdfium --extract-text --extract-tables --extract-charts --extract-infographics --text-depth page --vdb-upload --output-dir /home/local/jdyer/datasets/bo767-results-hf-ray