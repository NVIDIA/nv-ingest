# NeMo Retriever Extraction Client Examples

Contained in this repository are instructions to build a container with the NeMo Retriever extraction command line
interface (CLI) and Python API usage examples. Please follow the instructions below
to access guided examples.

## Step 0: Prerequisites
The system running the client does not require a GPU.

[Docker](https://docs.docker.com/get-started/get-docker/) - While the NeMo Retriever extraction client can
be installed directly on the host, this example creates a container with a few other add-ons.
Please follow the docker documentation for installation guidance.


## Step 1: Set some environment variables

```bash
# Used to mount external datasets to the container
DATASET_ROOT=...
# Used to connect to the cluster
REDIS_HOST=...
REDIS_PORT=...
TASK_QUEUE="morpheus_task_queue"
# Used to connect to minio object store, not needed if not using this option
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
```

## Step 2: Build the NeMo Retriever extraction client container

```bash
docker build -f docker/Dockerfile.client . -t nv-ingest-client
```

## Step 3: Run the container

The command below will start the container and expose a jupyter lab on port 8888.
Please modify the port mapping if you would like to expose jupyer on a different port.

```bash
docker run -it --rm \
    -v ${DATASET_ROOT}:/workspace/client_examples/data \
    -e REDIS_HOST=${REDIS_HOST} \
    -e REDIS_PORT=${REDIS_PORT} \
    -e TASK_QUEUE=${TASK_QUEUE} \
    -e MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY} \
    -e MINIO_SECRET_KEY=${MINIO_SECRET_KEY} \
    -p 8888:8888 \
    --name nv-ingest-client \
    nv-ingest-client:latest bash
```

## Step 4: Explore the examples

There are two usage examples included:

- Illustrates usage of the NeMo Retriever extraction client CLI:

    `/workspace/client_examples/examples/cli_client_usage.ipynb`

- Illustrates usage of the NeMo Retriever extraction client Python API:

    `/workspace/client_examples/examples/python_client_usage.ipynb`

Note, the default examples use sample data files provided in the NeMo Retriever extraction GitHub project. You will need to change these output indices to work with custom pdf source files.
