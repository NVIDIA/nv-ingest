# NVIDIA Ingest 26.1.0

## New Features

- **Parallel PDF Processing (V2 API):** Large PDFs are now automatically split into chunks and processed in parallel, delivering 2–9x faster ingestion for long documents depending on content. The V2 API also resolves historical processing issues with very large files while maintaining extraction quality. The response format remains backwards-compatible and can be enablde with `message_client_kwargs={"api_version": "v2"}`. V2 is now the default processing pipeline. See the [V2 API Guide](docs/docs/extraction/v2-api-guide.md).

## Improvements

- Added `extract_text` and `extract_images` parameter defaults (True) to the Ingestor extract method for consistency with `extract_tables` and `extract_charts`
- Added VLM caption prompt customization with `caption_prompt` and `caption_system_prompt` parameters, including reasoning control via `/think` and `/no_think` system prompts

## Bug Fixes

- ...
