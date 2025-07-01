# LLM Text Splitter Stage

The `LLMTextSplitterStage` is an advanced pipeline stage designed to split large documents into smaller, more manageable chunks while preserving the semantic structure of the original document. It uses Markdown headers as the primary splitting points and can optionally leverage a Large Language Model (LLM) for more nuanced splitting of text sections that are still too large.

## Key Features

- **Structural Splitting**: Intelligently splits documents based on Markdown headers (e.g., `#`, `##`, `###`).
- **Hierarchical Metadata**: Generates a `hierarchical_header` in the metadata for each chunk, representing its position in the document's structure (e.g., `Chapter 1 > Section 2 > Subsection A`).
- **LLM-Powered Sub-splitting**: For text chunks that still exceed the specified size after the initial Markdown split, the stage can use an LLM to find the most logical break-point within the text.
- **Highly Configurable**: All aspects of the splitter, from chunk size to LLM endpoints, can be configured.

## How to Use

To add the LLM Text Splitter to your pipeline, use the `add_llm_text_splitter_stage` helper function from the stage builders.

```python
from nv_ingest.framework.orchestration.ray.util.pipeline.stage_builders import add_llm_text_splitter_stage

# Within your pipeline definition...
add_llm_text_splitter_stage(pipeline, default_cpu_count)
```

This stage is activated for any job that includes `"split"` in its list of tasks.

## Configuration

The stage is configured primarily through environment variables.

| Environment Variable | Default Value | Description |
| --- | --- | --- |
| `LLM_SPLITTER_NIM_ENDPOINT` | `https://integrate.api.nvidia.com/v1` | The NVIDIA NIM endpoint for the LLM service. Required if `subsplit_with_llm` is enabled. |
| `LLM_SPLITTER_MODEL_NAME` | `meta/llama-3.1-8b-instruct` | The name of the model to use for finding split points. |
| `LLM_SPLITTER_API_KEY_ENV_VAR` | `NVIDIA_API_KEY` | The name of the environment variable that holds the API key for the LLM service. |

### Advanced Configuration

You can also override the default splitting behavior (e.g., `chunk_size`, `chunk_overlap`, `subsplit_with_llm`) by passing a custom `LLMTextSplitterSchema` object when adding the stage, or by providing task-level configuration overrides in the job submission.

### Performance Note

Enabling `subsplit_with_llm` can significantly increase the processing time and cost for large documents, as it involves making network calls to an external LLM service for each oversized chunk. It is recommended to use this feature judiciously. 