import os
import time
import logging
import threading
import multiprocessing

from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from nv_ingest.util.pipeline.pipeline_runners import run_ingest_pipeline

logger = logging.getLogger(__name__)


def worker_loop():
    time.sleep(20)

    # Set up the NvIngestClient and Ingestor inside the worker loop
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost"
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
        ).split(
            split_by="word",
            split_length=300,
            split_overlap=10,
            max_character_length=5000,
            sentence_window_size=0,
        )
    )

    # We'll run ingest in a separate thread to allow printing progress
    def run_ingest():
        try:
            results = ingestor.ingest()
            logger.info("Ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

    ingest_thread = threading.Thread(target=run_ingest)
    ingest_thread.start()

    # Print "." every 5 seconds until ingestion finishes
    while ingest_thread.is_alive():
        print(".", end="", flush=True)
        time.sleep(5)

    ingest_thread.join()
    print("\nIngest done.")


if (__name__ == "__main__"):
    # Set environment variables
    os.environ["CACHED_GRPC_ENDPOINT"] = "localhost:8007"
    os.environ["CACHED_INFER_PROTOCOL"] = "grpc"
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
    os.environ["PADDLE_GRPC_ENDPOINT"] = "localhost:8010"
    os.environ["PADDLE_HTTP_ENDPOINT"] = "http://localhost:8009/v1/infer"
    os.environ["PADDLE_INFER_PROTOCOL"] = "grpc"
    os.environ["REDIS_MORPHEUS_TASK_QUEUE"] = "morpheus_task_queue"
    os.environ["YOLOX_INFER_PROTOCOL"] = "grpc"
    os.environ["YOLOX_GRPC_ENDPOINT"] = "localhost:8001"
    os.environ[
        "VLM_CAPTION_ENDPOINT"] = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

    # Start the print/ingestor loop in a subprocess
    client_process = multiprocessing.Process(target=worker_loop)
    client_process.start()

    # Start the pipeline (this presumably blocks until pipeline completes)
    run_ingest_pipeline()

    # Once the pipeline is done, terminate and join the print process if it's still running
    if client_process.is_alive():
        client_process.terminate()
    client_process.join()
