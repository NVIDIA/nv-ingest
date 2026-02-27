# Add User-defined Stages to Your NeMo Retriever Extraction Pipeline

This documentation demonstrates how to add user-defined stages to your [NeMo Retriever extraction](overview.md) pipeline.
You can directly import a function, or use a string module path, and include robust signature validation. 
By following these steps, 
your Lambda stages are robust, signature-validated, plug-and-play for your RayPipeline, 
and operate on a well-defined DataFrame payload and metadata structure.

!!! note

    NeMo Retriever extraction is part of the NeMo Retriever Library.


To add user-defined stages to your pipeline, you need the following:

- **A callable function** — Your function must have the following exact signature. For more information, refer to []().
  
    ```python
    def my_fn(control_message: IngestControlMessage, stage_config: MyConfig) -> IngestControlMessage:
    ```

- **A DataFrame payload** — The `control_message.payload` field must be a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html). For more information, refer to [Create a DataFrame Payload](#create-a-dataframe-payload).

- **Valid metadata** — The `metadata` field must conform to the [NeMo Retriever metadata schema](content-metadata.md). For more information, refer to [Update and Validate Metadata](#update-and-validate-metadata).



## Create a Lambda Function and Config

Your function must have the following exact signature. 

```python
def my_fn(control_message: IngestControlMessage, stage_config: MyConfig) -> IngestControlMessage:
...
```

- The first parameter is named `control_message` and is an `IngestControlMessage`.
- The second parameter is named `stage_config` and must be a subclass of `pydantic.BaseModel`.
- The return value is an `IngestControlMessage`.


The following example demonstrates how to create a valid Lambda function and configuration.

```python
import pandas as pd
from pydantic import BaseModel
from nemo_retriever.internal.primitives.ingest_control_message import IngestControlMessage
from nemo_retriever.internal.schemas.meta.metadata_schema import validate_metadata

# Config schema for your stage
class MyToyConfig(BaseModel):
  set_processed: bool = True

def toy_stage_fn(control_message: IngestControlMessage, stage_config: MyToyConfig) -> IngestControlMessage:
  df = control_message.payload()

  # Set 'processed' flag in the allowed 'content_metadata' dict (if present)
  def update_metadata(meta):
    meta = dict(meta)

    # Only update if 'content_metadata' exists and is a dict
    if "content_metadata" in meta and isinstance(meta["content_metadata"], dict):
      meta["content_metadata"] = dict(meta["content_metadata"])  # ensure copy
      meta["content_metadata"]["processed"] = stage_config.set_processed
    validate_metadata(meta)
    return meta

  df["metadata"] = df["metadata"].apply(update_metadata)
  control_message.payload(df)
  return control_message
```



## Create a DataFrame Payload

The `control_message.payload` field must be a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with the following columns.

- `document_type`
- `source_id`
- `job_id`
- `metadata`

The following example demonstrates an input payload (before the stage runs) as a DataFrame with valid metadata.

```python
import pandas as pd

df = pd.DataFrame([
  {
    "document_type": "invoice",
    "source_id": "A123",
    "job_id": "job-001",
    "metadata": {
      "content": "example",
      "content_metadata": {"type": "pdf"},
      "source_metadata": {"source_id": "A123", "source_type": "pdf"},
    },
  },
  {
    "document_type": "report",
    "source_id": "B456",
    "job_id": "job-002",
    "metadata": {
      "content": "another",
      "content_metadata": {"type": "pdf"},
      "source_metadata": {"source_id": "B456", "source_type": "pdf"},
    }
  }
])
```



## Add a Stage (Function is Imported)

If your function is already imported in Python, 
use the following code to add your user-defined stage to the pipeline.

```python
config = MyToyConfig(flag_field="my_flag")

pipeline.add_stage(
  name="toy_stage",
  stage_actor=toy_stage_fn,
  config=config,
  min_replicas=1,
  max_replicas=2,
)
```



## Add a Stage (Function is Defined in a Module)

If your function is defined in a module, 
use the following code to add your user-defined stage to the pipeline. 
In this example, the function is defined in `my_project.stages:toy_stage_fn`.

```python
config = MyToyConfig(flag_field="has_been_processed")

pipeline.add_stage(
  name="toy_stage",
  stage_actor="my_project.stages:toy_stage_fn",
  config=config,
  min_replicas=1,
  max_replicas=2,
)
```

When the pipeline runs it does the following:

- Import and validate the function (using `resolve_callable_from_path`).
- Automatically wrap it as a Ray stage.
- Enforce the signature and parameter naming rules.



## Update and Validate Metadata

The `metadata` column in each row is a dictionary (JSON object), 
and must conform to the [NeMo Retriever metadata schema](content-metadata.md). 

After you change any metadata, you can validate it by using the `validate_metadata` function 
as demonstrated in the following code example.

```python
from nemo_retriever.internal.schemas.meta.metadata_schema import validate_metadata

def edit_metadata(control_message: IngestControlMessage, stage_config: MyToyConfig) -> IngestControlMessage:
  df = control_message.payload()

  def ensure_valid(meta):
    meta = dict(meta)
    # Only update an allowed nested metadata field
    if "content_metadata" in meta and isinstance(meta["content_metadata"], dict):
      meta["content_metadata"] = dict(meta["content_metadata"])
      meta["content_metadata"]["checked"] = True
    validate_metadata(meta)
    return meta

  df["metadata"] = df["metadata"].apply(ensure_valid)
  control_message.payload(df)
  return control_message
```




## Troubleshoot Validation Failures

The following are some examples of reasons that Lambda functions are invalid and fail validation.


### Wrong parameter names

The following Lambda function fails validation because the parameter names are incorrect. 
You should see an error message similar to `TypeError: Expected parameter names: 'control_message', 'config'`.

```python
#Incorrect example, do not use
def bad_fn(msg: IngestControlMessage, cfg: MyToyConfig) -> IngestControlMessage:
...
```

### Missing type annotations

The following Lambda function fails validation because the parameter and return types are missing.
You should see an error message similar to `TypeError`.

```python
#Incorrect example, do not use
def bad_fn(control_message, stage_config):
...
```

### Best Practices

Use the following best practices to avoid validation failures.

- Always use explicit type annotations and the required parameter names (`control_message`, `stage_config`).
- Your config can be any subclass of `pydantic.BaseModel`.
- Any errors in signature validation are raised with a clear message during pipeline construction.
- You can use `validate_metadata(meta)` to assert compliance after metadata changes.



## Minimal Complete Example

The  following example adds user-defined stages to your NeMo Retriever extraction pipeline. 

1. The following code creates a function for a user-defined stage.

    ```python
    # my_pipeline/stages.py
    from pydantic import BaseModel
    from nemo_retriever.internal.primitives.ingest_control_message import IngestControlMessage
    from nemo_retriever.internal.schemas.meta.metadata_schema import validate_metadata

    class DoubleConfig(BaseModel):
    multiply_by: int = 2

    def double_amount(control_message: IngestControlMessage, stage_config: DoubleConfig) -> IngestControlMessage:
    df = control_message.payload()
    
    # Suppose the metadata for each row includes 'amount' under 'content_metadata'
    def double_meta(meta):
        meta = dict(meta)
        if "content_metadata" in meta and isinstance(meta["content_metadata"], dict):
        cm = dict(meta["content_metadata"])
        if "amount" in cm and isinstance(cm["amount"], (int, float)):
            cm["amount"] *= stage_config.multiply_by
        meta["content_metadata"] = cm
        validate_metadata(meta)
        return meta

    df["metadata"] = df["metadata"].apply(double_meta)
    control_message.payload(df)
    return control_message
    ```

2. The following code adds the user-defined stage to the pipeline.

    - (Option 1)  For a function that is defined in the module my_pipeline.stages.

        ```python
        from my_pipeline.stages import double_amount, DoubleConfig

        pipeline.add_stage(
        name="doubler",
        stage_actor="my_pipeline.stages:double_amount",
        config=DoubleConfig(multiply_by=3),
        min_replicas=1,
        max_replicas=2,
        )
        ```

    - (Option 2) For a function that you have already imported.

        ```python
        pipeline.add_stage(
        name="doubler",
        stage_actor=double_amount,
        config=DoubleConfig(multiply_by=3),
        min_replicas=1,
        max_replicas=2,
        )
        ```



## Related Topics

- [User-Defined Functions for NeMo Retriever Extraction](user-defined-functions.md)
- [NimClient Usage](nimclient.md)
