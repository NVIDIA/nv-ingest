# User defined stages in NV-Ingest 

This guide shows how to add user defined stages in your NV-Ingest deployment using either a directly imported function or a string module path, with robust signature validation.

## Prerequisites

- **Callable signature:** Your function must have this exact signature:
  ```python
  def my_fn(control_message: IngestControlMessage, stage_config: MyConfig) -> IngestControlMessage:
  ...
  ```
  - The first parameter must be named **`control_message`** and annotated with `IngestControlMessage`.
  - The second parameter must be named **`stage_config`** and annotated with a subclass of `pydantic.BaseModel`.
  - The return value must be annotated as `IngestControlMessage`.

- **Payload/DataFrame Requirements:**
  - The control message's payload (`control_message.payload()`) will always be a **pandas DataFrame**.
  - The DataFrame will always have the following columns:
    - `document_type`
    - `source_id`
    - `job_id`
    - `metadata`
  - `metadata` is a nested JSON object (dict) for each row, and **must conform to the NV-Ingest Metadata Schema**.
    If you want to verify compliance after modifying metadata, use:
    ```python
    from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
    ...
      validate_metadata(metadata)  # raises on failure
    ```

---

## 1. Example: Creating a Valid Lambda Function and Config

```python
import pandas as pd
from pydantic import BaseModel
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata

# Config schema for your stage
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

---

## 2. DataFrame Example

Your input payload (before the stage runs) might look like:

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

---

## 3. Adding a Stage with a Direct Callable

If your function is **already imported in Python**:

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

---

## 4. Adding a Stage by String Specification

If your function is **defined in a module** (e.g., `my_project.stages:toy_stage_fn`):

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

The pipeline will:
- Import and validate the function (using `resolve_callable_from_path`).
- Automatically wrap it as a Ray stage.
- Enforce the signature and parameter naming rules.

---

## 5. Working with Metadata

Each row’s `"metadata"` column contains a dictionary (JSON object) with fields as required by the NV-Ingest Metadata Schema.
**After any modification, you may validate a row’s metadata as:**

```python
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata

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

---

## 6. Example: Invalid Lambda Functions (Will Fail Validation)

**Wrong parameter names:**
```python
def bad_fn(msg: IngestControlMessage, cfg: MyToyConfig) -> IngestControlMessage:
...
# Will raise TypeError: Expected parameter names: 'control_message', 'config'
```

**Missing type annotations:**
```python
def bad_fn(control_message, stage_config):
...
# Will raise TypeError for missing type annotations
```

---

## 7. Tips

- Always use **explicit type annotations** and required parameter names (`control_message`, `stage_config`).
- Your config can be any subclass of `pydantic.BaseModel`.
- Any errors in signature validation will be raised with a clear message during pipeline construction.
- You can use `validate_metadata(meta)` to assert compliance after changes.

---

## 8. Minimal Complete Example

Suppose your function lives in `my_pipeline.stages`:

```python
# my_pipeline/stages.py
from pydantic import BaseModel
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata

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

Then use:

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

or, if already imported:

```python
pipeline.add_stage(
  name="doubler",
  stage_actor=double_amount,
  config=DoubleConfig(multiply_by=3),
  min_replicas=1,
  max_replicas=2,
)
```


**That’s it! Lambda stages are now robust, signature-validated, plug-and-play for your RayPipeline, and operate on a well-defined DataFrame payload and metadata structure.**
