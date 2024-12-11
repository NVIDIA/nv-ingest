import json
import logging
import os
import sys
import time

import click

from nv_ingest.schemas import PipelineConfigSchema, ImageCaptionExtractionSchema, PDFExtractorSchema
from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.pdf_extractor_schema import PDFiumConfigSchema
from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema, TableExtractorConfigSchema
from nv_ingest.util.pipeline.pipeline_runners import start_pipeline_subprocess
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_client.message_clients.simple.simple_client import SimpleClient
from perf_pipeline import pipeline

logger = logging.getLogger(__name__)


def run_ingestor(config: PipelineConfigSchema):
    """
    Set up and run the ingestion process to send traffic against the pipeline.

    Parameters
    ----------
    config : PipelineConfigSchema
        The pipeline configuration schema.
    """
    logger.info("Setting up Ingestor client...")

    # The ingestor references environment variables internally, or we can pass the config values directly.
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=config.MESSAGE_CLIENT_PORT,
        message_client_hostname=config.MESSAGE_CLIENT_HOST,
    )

    # If needed, you can add logic to use config-derived paths here.
    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
        )
        .split(
            split_by="word",
            split_length=300,
            split_overlap=10,
            max_character_length=5000,
            sentence_window_size=0,
        )
    )

    try:
        results = ingestor.ingest()
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

    print("\nIngest done.")


def load_config(config_path: str = None) -> PipelineConfigSchema:
    """
    Load configuration from a file or use defaults from schemas.
    Environment variables are only used for explicitly supported parameters.

    Parameters
    ----------
    config_path : str, optional
        Path to a configuration file.

    Returns
    -------
    PipelineConfigSchema
        A fully validated configuration object.
    """
    if config_path:
        # Load configuration from a file (e.g., JSON)
        with open(config_path, "r") as f:
            data = json.load(f)
        return PipelineConfigSchema(**data)
    else:
        task_source_stage = {
            "source_path": "./data/multimodal_test.pdf",
            "source_type": "file",
        }

        embed_extractions_env = {
            "api_key": os.getenv("EMBED_EXTRACTIONS_API_KEY"),
            "embedding_nim_endpoint": os.getenv("EMBED_EXTRACTIONS_ENDPOINT"),
        }

        image_caption_extraction_env = {
            "api_key": os.getenv("IMAGE_CAPTION_EXTRACTION_API_KEY"),
        }

        pdfium_config_env = {
            "yolox_endpoints": (
                os.getenv("YOLOX_GRPC_ENDPOINT"),
                os.getenv("YOLOX_HTTP_ENDPOINT"),
            ),
            "yolox_infer_protocol": os.getenv("YOLOX_INFER_PROTOCOL"),
        }

        table_extractor_env = {
            "paddle_endpoints": (
                os.getenv("PADDLE_GRPC_ENDPOINT"),
                os.getenv("PADDLE_HTTP_ENDPOINT"),
            ),
            "paddle_infer_protocol": os.getenv("PADDLE_INFER_PROTOCOL"),
        }

        # Remove None values to allow schema defaults to apply
        embed_extractions_env = {k: v for k, v in embed_extractions_env.items() if v is not None}
        image_caption_extraction_env = {k: v for k, v in image_caption_extraction_env.items() if v is not None}
        pdfium_config_env = {k: v for k, v in pdfium_config_env.items() if v is not None}
        table_extractor_env = {k: v for k, v in table_extractor_env.items() if v is not None}

        return PipelineConfigSchema(
            embed_extractions_module=EmbedExtractionsSchema(**embed_extractions_env),
            image_caption_extraction_module=ImageCaptionExtractionSchema(**image_caption_extraction_env),
            pdf_extractor_module=PDFExtractorSchema(
                pdfium_config=PDFiumConfigSchema(**pdfium_config_env) if pdfium_config_env else None
            ),
            table_extractor_module=TableExtractorSchema(
                stage_config=TableExtractorConfigSchema(**table_extractor_env) if table_extractor_env else None
            ),
        )


@click.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="Path to config file.")
def main(config):
    """
    Main entry point for the script. Loads configuration and orchestrates
    the pipeline subprocess and ingestion process.

    Parameters
    ----------
    config : str, optional
        Path to a configuration file.
    """
    # Load configuration (either from file, environment, or defaults)
    pipeline_config = load_config(config)

    try:
        # Start the pipeline subprocess with the given config
        pipeline_process = start_pipeline_subprocess(pipeline_config)

        # Optionally, wait a bit before starting the ingestor
        time.sleep(10)

        # Run ingestion after starting the pipeline
        run_ingestor(pipeline_config)

        pipeline_process.wait()

        # The main program will exit, and the atexit handler will terminate the subprocess group
    except Exception as e:
        logger.error(f"Error running pipeline subprocess or ingestion: {e}")
        # The atexit handler will ensure subprocess termination
        sys.exit(1)
