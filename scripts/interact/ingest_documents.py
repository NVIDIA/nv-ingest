"""
Ingestion and indexing helper script.

This script also supports loading option values from a `.env` file located in
the same directory as this script. Environment variables with the `INGEST_`
prefix will be mapped to option names (e.g., `INGEST_HOSTNAME` -> `--hostname`).
Command-line arguments take precedence over environment variables.
"""

import time
from typing import Optional

import click

from nv_ingest_client.client import Ingestor

from utils import clean_spill, kv_event_log, _load_env_file_from_same_directory


@click.command(
    context_settings={
        "help_option_names": ["-h", "--help"],
        # Allow mapping INGEST_<OPTION_NAME> env vars to options automatically
        "auto_envvar_prefix": "INGEST",
    }
)
@click.option(
    "--hostname",
    default="localhost",
    show_default=True,
    help="""
    Hostname used by the message client for the `Ingestor`
    and, if `--milvus-uri` is not set, to construct the Milvus URI.
    """,
)
@click.option(
    "--input-glob",
    required=True,
    help="File glob specifying the input documents to ingest (e.g., PDFs).",
)
@click.option(
    "--extract-text/--no-extract-text",
    default=True,
    show_default=True,
    help="Whether to extract text from input documents.",
)
@click.option(
    "--extract-tables/--no-extract-tables",
    default=True,
    show_default=True,
    help="Whether to extract tables from input documents.",
)
@click.option(
    "--extract-charts/--no-extract-charts",
    default=True,
    show_default=True,
    help="Whether to extract charts from input documents.",
)
@click.option(
    "--extract-images/--no-extract-images",
    default=False,
    show_default=True,
    help="Whether to extract images from input documents.",
)
@click.option(
    "--text-depth",
    default="page",
    show_default=True,
    help="Granularity for text extraction (e.g., 'page').",
)
@click.option(
    "--table-output-format",
    default="markdown",
    show_default=True,
    help="Output format for extracted tables (e.g., 'markdown').",
)
@click.option(
    "--model-name",
    default=None,
    help="""
    Embedding model name to use during `.embed(...)`.
    If not provided, defaults to the value from `embed_info()` if available.
    """,
)
@click.option(
    "--spill-path",
    default="/tmp/spill",
    show_default=True,
    help="Directory used for saving intermediate results to disk.",
)
@click.option(
    "--pre-clean-spill/--no-pre-clean-spill",
    default=True,
    show_default=True,
    help="Clean the spill directory before ingestion starts.",
)
@click.option(
    "--post-clean-spill/--no-post-clean-spill",
    default=False,
    show_default=True,
    help="Clean the spill directory after processing completes.",
)
@click.option(
    "--show-progress/--no-show-progress",
    default=True,
    show_default=True,
    help="Display a progress bar during ingestion.",
)
@click.option(
    "--return-failures/--no-return-failures",
    default=True,
    show_default=True,
    help="Return a list of failures from the ingestion run.",
)
@click.option(
    "--save-to-disk/--no-save-to-disk",
    default=True,
    show_default=True,
    help="Persist intermediate results to disk at `--spill-path`.",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=False,
    show_default=True,
    help="If saving to disk, whether the `Ingestor` should clean up spill files it creates.",
)
@click.option(
    "--num-pages",
    type=int,
    default=54730,
    show_default=True,
    help="Total number of pages in the dataset (used only for throughput logging).",
)
@click.option(
    "--num-files",
    type=int,
    default=767,
    show_default=True,
    help="Total number of files in the dataset (used only for throughput logging).",
)
def main(
    hostname: str,
    input_glob: str,
    extract_text: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_images: bool,
    text_depth: str,
    table_output_format: str,
    model_name: Optional[str],
    spill_path: str,
    pre_clean_spill: bool,
    post_clean_spill: bool,
    show_progress: bool,
    return_failures: bool,
    save_to_disk: bool,
    cleanup: bool,
    num_pages: int,
    num_files: int,
):
    """Run ingestion, embedding, and indexing with configurable options."""

    # Ideaology here is that we want to be able to run the ingestion script multiple times
    # and don't want data from previous runs to be included in the next run.
    # This is why we clean the spill directory before each run.
    if pre_clean_spill:
        clean_spill(spill_path)

    # Setup tasks unrelated to runtime should be performed here before the ingest_start time is captured
    # TODO: Any setup you need ...

    ingestion_start = time.time()

    # Create Ingestor object by chaining together the various methods
    # that are enabled by the user.
    ingestor = Ingestor(message_client_hostname=hostname)
    ingestor = (
        Ingestor(message_client_hostname=hostname)
        .files(input_glob)
        .extract(
            extract_text=extract_text,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_images=extract_images,
            text_depth=text_depth,
            table_output_format=table_output_format,
        )
        .embed(model_name=model_name)
        .save_to_disk(output_directory=spill_path, cleanup=cleanup)
    )

    results, failures = ingestor.ingest(
        show_progress=show_progress, return_failures=return_failures, save_to_disk=save_to_disk
    )

    if return_failures and len(failures) > 0:
        print(f"WARNING: {len(failures)} failures occurred during ingestion.")
        for failure in failures:
            print(f"Failure: {failure}")

    ingestion_end = time.time()
    ingestion_time = ingestion_end - ingestion_start
    kv_event_log("ingestion_time", ingestion_time)
    if ingestion_time > 0:
        kv_event_log("ingestion_pages_per_sec", num_pages / ingestion_time)
        kv_event_log("ingestion_files_per_sec", num_files / ingestion_time)
    kv_event_log("failure_count", len(failures))
    kv_event_log("failures", failures)
    kv_event_log("record_count", len(results))

    # Default to (False) not cleaning the spill directory after the run.
    # This ensures that if you want to examine the data or chain
    # another script to the run you can do so.
    if post_clean_spill:
        clean_spill(spill_path)


if __name__ == "__main__":
    # Load `.env` before Click parses options so env vars can supply defaults
    _load_env_file_from_same_directory()

    main()
