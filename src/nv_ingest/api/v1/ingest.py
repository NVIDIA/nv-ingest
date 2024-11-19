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
from typing import Annotated, Dict, List, Optional
import base64
import json
import logging
import time
import traceback
import uuid
import os
import requests

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File, UploadFile, Form
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from nv_ingest_client.primitives.jobs.job_spec import JobSpec
from opentelemetry import trace
from redis import RedisError
from pymilvus import Collection, MilvusClient, DataType, connections, utility, BulkInsertState
from pymilvus.bulk_writer import RemoteBulkWriter, BulkFileType
import openai
from pydantic import BaseModel

from nv_ingest.util.converters.formats import ingest_json_results_to_blob
from nv_ingest.util.vdb.milvus import bulk_upload_results_to_milvus, search_milvus
from nv_ingest.schemas.vdb_query_job_schema import OpenAIRequest, OpenAIResponse

from nv_ingest_client.primitives.tasks.extract import ExtractTask
from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.schemas.processing_job_schema import ConversionStatus, ProcessingJob
from nv_ingest.service.impl.ingest.redis_ingest_service import RedisIngestService
from nv_ingest.service.meta.ingest.ingest_service_meta import IngestServiceMeta
from nv_ingest_client.primitives.tasks.table_extraction import TableExtractionTask
from nv_ingest_client.primitives.tasks.chart_extraction import ChartExtractionTask
from nv_ingest_client.primitives.tasks.vdb_upload import VdbUploadTask
from nv_ingest_client.primitives.tasks.embed import EmbedTask
from nv_ingest_client.primitives.tasks.split import SplitTask

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
async def convert_pdf(
    ingest_service: INGEST_SERVICE_T,
    files: List[UploadFile] = File(...),
    job_id: str = Form(...),
    vdb_task: bool = Form(False)
) -> Dict[str, str]:
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

            extract_task = ExtractTask(
                document_type="pdf",
                extract_text=True,
                extract_images=True,
                extract_tables=True
            )

            table_data_extract = TableExtractionTask()
            chart_data_extract = ChartExtractionTask()
            split_task = SplitTask(
                split_by="word",
                split_length=300,
                split_overlap=10,
                max_character_length=5000,
                sentence_window_size=0,
            )
            embed_task = EmbedTask(
                text=True,
                tables=True
            )

            job_spec.add_task(extract_task)
            job_spec.add_task(table_data_extract)
            job_spec.add_task(chart_data_extract)
            job_spec.add_task(split_task)
            job_spec.add_task(embed_task)

            submitted_job_id = await ingest_service.submit_job(
                MessageWrapper(
                    payload=json.dumps(job_spec.to_dict())
                )
            )

            processing_job = ProcessingJob(
                submitted_job_id=submitted_job_id,
                filename=file.filename,
                status=ConversionStatus.IN_PROGRESS,
                vdb_task=vdb_task
            )
            
            submitted_jobs.append(processing_job)

        # Each invocation of this endpoint creates a "job" that could have multiple PDFs being parsed ...
        # keep a local cache of those job ids to the submitted job ids so that they can all be checked later
        if job_id is None:
            job_id = str(uuid.uuid4())
        else:
            job_id = job_id

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
                blob_response = ingest_json_results_to_blob(job_response)
                
                processing_job.raw_result = job_response
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
        raw_results = []
        vdb_task = False
        for result in updated_cache:
            results.append(
                {
                    "filename": result.filename,
                    "status": "success",
                    "content": result.content,
                }
            )
            raw_results.append(result.raw_result)
            if result.vdb_task == True:
                vdb_task = True
        
        if vdb_task:
            print(f"Inserting results into Milvus vector database ...")
            resp = bulk_upload_results_to_milvus(raw_results)
        
        return JSONResponse(
            content={"status": "completed", "result": results},
            status_code=200,
        )
    else:
        # Not yet ready ...
        raise HTTPException(status_code=202, detail="Job is not ready yet. Retry later.")


def perform_text_embedding(text: str) -> List[float]:
    """Get the text embeddings use the nvcr.io/nim/nvidia/nv-embedqa-e5-v5 NIM"""
    try:
        embedding_http_endpoint = os.getenv("EMBEDDING_HTTP_ENDPOINT", "http://embedding:8000/v1/embeddings")
        logger.debug(f"Embedding endpoint: {embedding_http_endpoint}")
        
        # embedding JSON payload
        payload = {
            "input": [text],
            "model": "nvidia/nv-embedqa-e5-v5",
            "input_type": "query"
        }

        # Headers
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        
        result = requests.post(embedding_http_endpoint, data=json.dumps(payload), headers=headers)
        logger.debug(f"Embedding result: {result.text}")
        if result.status_code == 200:
            resp_json = json.loads(result.text)
            return resp_json['data'][0]['embedding'] # List of floats
        else:
            # Exception in embedding ...
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {result.status_code}")
            
    except Exception as e:
        logger.error(f"Exception embedding text: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

@router.post("/query", response_model=OpenAIResponse)
async def query_milvus(request: OpenAIRequest):
    
    # Adaptation for needs from another team. Sorry for the clutter
    model = "nvidia/nv-embedqa-e5-v5"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request.query}
    ]
    max_tokens = 100
    temperature = 0.7
    top_p = 1.0
    
    # Extract user query from the last message
    user_message = next(
        (message for message in reversed(messages) if message["role"] == "user"), None
    )
    if not user_message:
        raise HTTPException(status_code=400, detail="No user query message found in request.")

    logger.debug(f"Received user_message: {user_message}")
    user_query = user_message["content"]
    logger.debug(f"User query: {user_query}")
    
    # Generate embedding for the query
    query_embedding = perform_text_embedding(user_query)

    logger.debug(f"Query Embedding: {query_embedding}")
    # Query Milvus for relevant documents
    try:
        docs = search_milvus(query_embedding)
        logger.debug(f"Results from milvus: {docs}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus query failed: {e}")
    
    
    results = []
    # Construct the response (combine docs into one or summarize, depending on your application)
    if docs != None:
        print(f"Docs Type: {type(docs)}")
        print(f"Len of docs: {len(docs)}")
        
        for doc in docs:
            if doc is not None:
                print(f"Doc: {doc}")
                results.append(doc)
                if len(results) >= request.k:
                    break
            else:
                print(f"Doc is None ...")
            
    #     response_content = "\n".join([s for s in docs if s is not None])
    # else:
    #     # No hits from the VDB ....
    #     response_content = ""

    # return OpenAIResponse(
    #     id="query-response-001",
    #     object="chat.completion",
    #     created=int(time.time()),
    #     model="nvidia/nv-embedqa-e5-v5",
    #     choices=[{"message": {"role": "assistant", "content": response_content}}]
    # )
    
    return OpenAIResponse(
        content=results
    )
