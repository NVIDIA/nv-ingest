# Lambda Stages in NV-Ingest Pipelines: Dynamic Construction vs. Explicit Stage Classes

## Overview

In NV-Ingest, pipeline stages are usually implemented as classes derived from `RayActorStage` or its variants. This is the "explicit" approach, involving hand-written classes and decorated methods. While powerful, this pattern can become repetitive for simple, stateless operations.

The framework also supports "lambda stages": dynamic stage classes built using helpers like `wrap_callable_as_stage`, `wrap_callable_as_source`, and `wrap_callable_as_sink`. With this approach, a function and a Pydantic config schema are wrapped into a fully-featured Ray pipeline stage, including tracing and exception handling—no manual boilerplate required.

---

## Explicit Stage Definition

The traditional method for defining a stage looks like this:

```python
class TextSplitterStage(RayActorStage):
def __init__(self, config: TextSplitterSchema) -> None:
    super().__init__(config)
    self.validated_config = config

    @traceable("text_splitter")
    @filter_by_task(["split"])
    @nv_ingest_node_failure_try_except(annotation_id="text_splitter", raise_on_failure=False)
    def on_data(self, message: IngestControlMessage) -> IngestControlMessage:
        df_payload = message.payload()
        task_config = remove_task_by_type(message, "split")
        df_updated = transform_text_split_and_tokenize_internal(
            df_transform_ledger=df_payload,
            task_config=task_config,
            transform_config=self.validated_config,
            execution_trace_log=None,
        )
        message.payload(df_updated)
        return message
```

### Pros
- Maximum control and clarity
- Easy to add state, resources, or custom lifecycle logic

### Cons
- Repetitive boilerplate
- Requires manual consistency for decorators and error handling

---

## Lambda-Based Stage Definition (Dynamic Construction)

With lambda stages, you define a function and schema only. The framework generates a decorated stage class for you:

```python
def text_splitter_fn(message: IngestControlMessage, config: TextSplitterSchema) -> IngestControlMessage:
    df_payload = message.payload()
    task_config = remove_task_by_type(message, "split")
    df_updated = transform_text_split_and_tokenize_internal(
        df_transform_ledger=df_payload,
        task_config=task_config,
        transform_config=config,
        execution_trace_log=None,
    )
    message.payload(df_updated)
    
    return message

LambdaTextSplitterStage = wrap_callable_as_stage(
    text_splitter_fn,
    TextSplitterSchema,
    required_tasks=["split"],
    trace_id="text_splitter"
)
```

This `LambdaTextSplitterStage` is a drop-in replacement for the explicit class above.

### Usage Example

```python
pipeline.add_stage(
    name="text_splitter",
    stage_actor=LambdaTextSplitterStage,
    config=TextSplitterSchema(),
    min_replicas=0,
    max_replicas=2,
)
```

### Pros
- Concise: All boilerplate is handled for you
- Consistent: Tracing and exception-handling are always present
- Composable: Functions can be easily reused or composed

### Cons
- Less flexible for advanced custom logic (requires explicit class if needed)

---

## When Should You Use Lambda Stages?

**Lambda stages are ideal when:**
- Your logic can be described as a pure function
- You want fast prototyping with minimal code
- You don’t need per-actor state or complex lifecycle management

**Explicit classes are preferable when:**
- You need to manage state/resources beyond the config
- You need advanced customization

---

## Summary Table

|                        | Explicit Stage Class    | Lambda Stage (Dynamic)     |
|------------------------|:----------------------:|:--------------------------:|
| **Clarity**            | Very high              | High (for simple stages)   |
| **Boilerplate**        | High                   | Minimal                   |
| **Customization**      | Unlimited              | Limited to function logic |
| **Consistency**        | Manual                 | Automatic                 |
| **When to use**        | Advanced/custom needs  | Stateless, simple logic   |

---

## Conclusion

Lambda stages are a fast, clean alternative to explicit stage classes in NV-Ingest. They reduce boilerplate and make your pipelines more maintainable, while keeping the option to define full classes when you need advanced features.

---

**See also:**
- `wrap_callable_as_stage` docs
- NV-Ingest pipeline guide
