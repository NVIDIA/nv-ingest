# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import threading
import multiprocessing

from nv_ingest.util.pipeline.pipeline_runners import run_ingest_pipeline

# Set environment variables
os.environ["CACHED_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/nvdev/cv/university-at-buffalo/cached"
os.environ["CACHED_INFER_PROTOCOL"] = "http"
os.environ["DEPLOT_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/nvdev/vlm/google/deplot"
os.environ["DEPLOT_INFER_PROTOCOL"] = "http"
os.environ["INGEST_LOG_LEVEL"] = "DEBUG"
os.environ["MESSAGE_CLIENT_HOST"] = "localhost"
os.environ["MESSAGE_CLIENT_PORT"] = "7671"
os.environ["MESSAGE_CLIENT_TYPE"] = "simple"
os.environ["MINIO_BUCKET"] = "nv-ingest"
os.environ["MRC_IGNORE_NUMA_CHECK"] = "1"
# os.environ["NGC_API_KEY"] = "NGC_API_KEY"
# os.environ["NVIDIA_BUILD_API_KEY"] = "YOUR_API_KEY"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "otel-collector:4317"
os.environ["PADDLE_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/nvdev/cv/baidu/paddleocr"
os.environ["PADDLE_INFER_PROTOCOL"] = "http"
os.environ["REDIS_MORPHEUS_TASK_QUEUE"] = "morpheus_task_queue"
os.environ["YOLOX_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_HTTP_ENDPOINT"] = "http://localhost:8000/v1/infer"
os.environ[
    "VLM_CAPTION_ENDPOINT"] = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"


def worker_loop():
    while True:
        time.sleep(5)
        print(".")


if __name__ == "__main__":
    # Start the print loop in a subprocess
    print_process = multiprocessing.Process(target=worker_loop)
    print_process.start()

    run_ingest_pipeline()

    # Once the pipeline is done, terminate the print loop process
    print_process.terminate()
    print_process.join()
