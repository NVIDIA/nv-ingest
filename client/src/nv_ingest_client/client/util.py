import logging
import time
from typing import List

from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from nv_ingest_client.util.file_processing.extract import extract_file_content


logger = logging.getLogger(__name__)


def create_job_specs_for_batch(files_batch: List[str]) -> List[JobSpec]:
    job_specs = []
    for file_name in files_batch:
        try:
            file_content, file_type = extract_file_content(file_name)  # Assume these are defined
            file_type = file_type.value
        except ValueError as ve:
            logger.error(f"Error extracting content from {file_name}: {ve}")
            continue

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=file_name,
            source_name=file_name,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )
        job_specs.append(job_spec)

    return job_specs
