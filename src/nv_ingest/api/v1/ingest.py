# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pylint: skip-file

from io import BytesIO
from typing import Annotated, Dict, List
import base64
import json
import logging
import time
import traceback
import uuid

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File, UploadFile
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from opentelemetry import trace
from redis import RedisError
from pymilvus import Collection, connections, utility
import openai

from nv_ingest_client.primitives.tasks.extract import ExtractTask
from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.schemas.processing_job_schema import ConversionStatus, ProcessingJob
from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask
from nv_ingest_client.primitives.tasks.vdb_upload import VdbUploadTask

logger = logging.getLogger("uvicorn")
tracer = trace.get_tracer(__name__)

router = APIRouter()


async def _get_ingest_service() -> IngestServiceMeta:
    """
    Gather the appropriate Ingestion Service to use for the nv-ingest endpoint.
    """
    logger.debug("Creating RedisIngestService singleton for dependency injection")
    return RedisIngestService.getInstance()


INGEST_SERVICE_T = Annotated[IngestServiceMeta, Depends(_get_ingest_service)]


# POST /submit
@router.post(
    "/submit",
    responses={
        200: {"description": "Submission was successful"},
        500: {"description": "Error encountered during submission"},
    },
    tags=["Ingestion"],
    summary="submit document to the core nv ingestion service for processing",
    operation_id="submit",
)
async def submit_job_curl_friendly(
    ingest_service: INGEST_SERVICE_T,
    file: UploadFile = File(...)
):
    """
    A multipart/form-data friendly Job submission endpoint that makes interacting with
    the nv-ingest service through tools like Curl easier.
    """
    try:
        file_stream = BytesIO(file.file.read())
        doc_content = base64.b64encode(file_stream.read()).decode("utf-8")

        # Construct the JobSpec from the HTTP supplied form-data
        job_spec = JobSpec(
            # TOOD: Update this to look at the uploaded content-type, currently that is not working
            document_type="pdf",
            payload=doc_content,
            source_id=file.filename,
            source_name=file.filename,
            # TODO: Update this to accept user defined options
            extended_options={
                "tracing_options":
                {
                    "trace": True,
                    "ts_send": time.time_ns(),
                    "trace_id": trace.get_current_span().get_span_context().trace_id
                }
            }
        )

        # This is the "easy submission path" just default to extracting everything
        extract_task = ExtractTask(
            document_type="pdf",
            extract_text=True,
            extract_images=True,
            extract_tables=True
        )

        job_spec.add_task(extract_task)

        submitted_job_id = await ingest_service.submit_job(
            MessageWrapper(
                payload=json.dumps(job_spec.to_dict())
            )
        )
        return submitted_job_id
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")

# POST /submit_job
@router.post(
    "/submit_job",
    responses={
        200: {"description": "Jobs were successfully submitted"},
        500: {"description": "Error encountered while submitting jobs."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="submit jobs to the core nv ingestion service for processing",
    operation_id="submit_job",
)
async def submit_job(job_spec: MessageWrapper, ingest_service: INGEST_SERVICE_T):
    try:
        # Inject the x-trace-id into the JobSpec definition so that OpenTelemetry
        # will be able to trace across uvicorn -> morpheus
        current_trace_id = trace.get_current_span().get_span_context().trace_id
        
        job_spec_dict = json.loads(job_spec.payload)
        job_spec_dict['tracing_options']['trace_id'] = current_trace_id
        updated_job_spec = MessageWrapper(
            payload=json.dumps(job_spec_dict)
        )

        submitted_job_id = await ingest_service.submit_job(updated_job_spec)
        return submitted_job_id
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")


# GET /fetch_job
@router.get(
    "/fetch_job/{job_id}",
    responses={
        200: {"description": "Job was successfully retrieved."},
        202: {"description": "Job is not ready yet. Retry later."},
        500: {"description": "Error encountered while fetching job."},
        503: {"description": "Service unavailable."},
    },
    tags=["Ingestion"],
    summary="Fetch a previously submitted job from the ingestion service by providing its job_id",
    operation_id="fetch_job",
)
async def fetch_job(job_id: str, ingest_service: INGEST_SERVICE_T):
    try:
        # Attempt to fetch the job from the ingest service
        job_response = await ingest_service.fetch_job(job_id)
        return job_response
    except TimeoutError:
        # Return a 202 Accepted if the job is not ready yet
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")
    except RedisError:
        # Return a 202 Accepted if the job could not be fetched due to Redis error, indicating a retry might succeed
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")
    except ValueError as ve:
        # Return a 500 Internal Server Error for ValueErrors
        raise HTTPException(status_code=500, detail=f"Value error encountered: {str(ve)}")
    except Exception as ex:
        # Catch-all for other exceptions, returning a 500 Internal Server Error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Nv-Ingest Internal Server Error: {str(ex)}")



@router.post("/convert")
async def convert_pdf(ingest_service: INGEST_SERVICE_T, files: List[UploadFile] = File(...)) -> Dict[str, str]:
    try:
        
        print(f"Processing: {len(files)} PDFs ....")
        
        submitted_jobs: List[ProcessingJob] = []
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(

                    status_code=400, detail=f"File {files[0].filename} must be a PDF"
                )
            
            file_stream = BytesIO(file.file.read())
            doc_content = base64.b64encode(file_stream.read()).decode("utf-8")
            
            job_spec = JobSpec(
                document_type="pdf",
                payload=doc_content,
                source_id=file.filename,
                source_name=file.filename,
                extended_options={
                    "tracing_options":
                    {
                        "trace": True,
                        "ts_send": time.time_ns(),
                    }
                }
            )

            # This is the "easy submission path" just default to extracting everything
            extract_task = ExtractTask(
                document_type="pdf",
                extract_text=True,
                extract_images=True,
                extract_tables=True
            )

            table_data_extract = TableExtractionTask()
            chart_data_extract = ChartExtractionTask()
            vdb_task = VdbUploadTask()

            job_spec.add_task(extract_task)
            job_spec.add_task(table_data_extract)
            job_spec.add_task(chart_data_extract)
            job_spec.add_task(vdb_task)

            submitted_job_id = await ingest_service.submit_job(
                MessageWrapper(
                    payload=json.dumps(job_spec.to_dict())
                )
            )
            
            processing_job = ProcessingJob(submitted_job_id=submitted_job_id, filename=file.filename, status=ConversionStatus.IN_PROGRESS)
            
            submitted_jobs.append(processing_job)

        # Each invocation of this endpoint creates a "job" that could have multiple PDFs being parsed ...
        # keep a local cache of those job ids to the submitted job ids so that they can all be checked later
        job_id = str(uuid.uuid4())
        await ingest_service.set_processing_cache(job_id, submitted_jobs) 
        
        print(f"Submitted: {len(submitted_jobs)} PDFs for processing")

        return {
            "task_id": job_id,
            "status": "processing",
            "status_url": f"/status/{job_id}",
        }

    except Exception as e:
        logger.error(f"Error starting conversion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
def parse_json_string_to_blob(json_content):
    """
    Parse a JSON string or BytesIO object, combine and sort entries, and create a blob string.

    Args:
        json_content (str or BytesIO): JSON string or BytesIO object containing JSON data.

    Returns:
        str: The generated blob string.
    """
    try:
        # Load the JSON data
        data = json.loads(json_content) if isinstance(json_content, str) else json.loads(json_content)
        data = data['data']

        # Smarter sorting: by page, then structured objects by x0, y0
        def sorting_key(entry):
            page = entry['metadata']['content_metadata']['page_number']
            if entry['document_type'] == 'structured':
                # Use table location's x0 and y0 as secondary keys
                x0 = entry['metadata']['table_metadata']['table_location'][0]
                y0 = entry['metadata']['table_metadata']['table_location'][1]
            else:
                # Non-structured objects are sorted after structured ones
                x0 = float('inf')
                y0 = float('inf')
            return page, x0, y0

        data.sort(key=sorting_key)

        # Initialize the blob string
        blob = []

        for entry in data:
            document_type = entry.get('document_type', '')

            if document_type == 'structured':
                # Add table content to the blob
                blob.append(entry['metadata']['table_metadata']['table_content'])
                blob.append("\n")

            elif document_type == 'text':
                # Add content to the blob
                blob.append(entry['metadata']['content'])
                blob.append("\n")

            elif document_type == 'image':
                # Add image caption to the blob
                caption = entry['metadata']['image_metadata'].get('caption', '')
                blob.append(f"image_caption:[{caption}]")
                blob.append("\n")

        # Join all parts of the blob into a single string
        return ''.join(blob)

    except Exception as e:
        print(f"[ERROR] An error occurred while processing JSON content: {e}")
        return ""


@router.get("/status/{job_id}")
async def get_status(ingest_service: INGEST_SERVICE_T, job_id: str):
    try:
        processing_jobs = await ingest_service.get_processing_cache(job_id)
    # TO DO: return 400 when job_id dne
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    print(f"Job contains: {len(processing_jobs)} PDFs that are processing. Lets check them now ...")
    
    updated_cache: List[ProcessingJob] = []
    num_ready_docs = 0
    
    for processing_job in processing_jobs:
        print(f"Checking on submitted_job_id: {processing_job.submitted_job_id}")
    
        print(f"Processing Job Status: {processing_job.status}")
        if processing_job.status == ConversionStatus.IN_PROGRESS:
            # Attempt to fetch the job from the ingest service
            try:
                job_response = await ingest_service.fetch_job(processing_job.submitted_job_id)
                
                job_response = json.dumps(job_response)
                blob_response = parse_json_string_to_blob(job_response)
                
                processing_job.content = blob_response
                processing_job.status = ConversionStatus.SUCCESS
                num_ready_docs = num_ready_docs + 1
                updated_cache.append(processing_job)
                
            except TimeoutError:
                print(f"TimeoutError getting result for job_id: {processing_job.submitted_job_id}")
                updated_cache.append(processing_job)
                continue
            except RedisError:
                print(f"RedisError getting result for job_id: {processing_job.submitted_job_id}")
                updated_cache.append(processing_job)
                continue
        else:
            print(f"{processing_job.submitted_job_id} has already finished successfully ....")
            num_ready_docs = num_ready_docs + 1
            updated_cache.append(processing_job)

    await ingest_service.set_processing_cache(job_id, updated_cache) 

    print(f"{num_ready_docs}/{len(updated_cache)} complete")
    if num_ready_docs == len(updated_cache):
        results = []
        for result in updated_cache:
            results.append(
                {
                    "filename": result.filename,
                    "status": "success",
                    "content": result.content,
                }
            )
        
        return JSONResponse(
            content={"status": "completed", "result": results},
            status_code=200,
        )
    else:
        # Not yet ready ...
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")


# Connect to Milvus
connections.connect("default", host="milvus", port="19530")  # Update with your Milvus host/port
COLLECTION_NAME = "your_collection_name"  # Update to your Milvus collection name

# Define request schema
class OpenAIRequest(BaseModel):
    model: str
    messages: List[dict]  # [{"role": "user", "content": "question"}]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0

# OpenAI-compatible response schema
class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]  # [{"message": {"role": "assistant", "content": "answer"}}]

def search_milvus(question_embedding: List[float], top_k: int = 5):
    """Query Milvus for nearest neighbors of the question embedding."""
    if not utility.has_collection(COLLECTION_NAME):
        raise HTTPException(status_code=500, detail=f"Milvus collection '{COLLECTION_NAME}' not found.")
    
    collection = Collection(COLLECTION_NAME)
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",  # Vector field in your collection
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content"]  # Update based on your schema
    )
    return [hit.entity.get("content") for hit in results[0]]

def embed_text_with_openai(text: str) -> List[float]:
    """Get text embeddings using OpenAI embeddings API."""
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

@app.post("/query", response_model=OpenAIResponse)
async def query_milvus(request: OpenAIRequest):
    # Extract user query from the last message
    user_message = next(
        (message for message in reversed(request.messages) if message["role"] == "user"), None
    )
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in request.")

    user_query = user_message["content"]
    
    # Generate embedding for the query
    query_embedding = embed_text_with_openai(user_query)

    # Query Milvus for relevant documents
    try:
        docs = search_milvus(query_embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")
    
    # Construct the response (combine docs into one or summarize, depending on your application)
    response_content = "\n".join(docs)

    return OpenAIResponse(
        id="query-response-001",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[{"message": {"role": "assistant", "content": response_content}}]
    )
