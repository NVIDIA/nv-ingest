# Contributing to NV-Ingest

External contributions to NV-Ingest will be welcome soon, and they are greatly appreciated! 
For more information, refer to [Contributing to NV-Ingest](https://github.com/NVIDIA/nv-ingest/blob/main/CONTRIBUTING.md).

## Documentation maintenance

Prefer **centralization over repetition** for:

- **Model names, tokenizer/config defaults, environment variables, build-time flags, and job schema fields** — keep one canonical description and link from elsewhere.
- **Architecture diagrams and descriptions of service interactions** — one canonical overview; other pages summarize or link.
- **Constraints, caveats, and “gotchas” that may change over time** — e.g. known issues in [Release Notes](releasenotes-nv-ingest.md), hardware/feature limits in [Support Matrix](support-matrix.md).

When adding or updating information, always ask:

1. **Where should the canonical description live?**  
   Use or create the appropriate reference page (see below).
2. **Are there other places that already mention this and should now link to or reference the canonical description instead?**  
   Update them to point to the single source of truth.

**Canonical reference pages:**

| Topic | Canonical page |
|-------|----------------|
| Product naming | [What is NeMo Retriever Extraction?](overview.md) |
| Environment variables | [Environment Variables](environment-config.md) |
| Pipeline scaling env vars | [Resource Scaling Modes](scaling-modes.md) |
| Tokenizer and split defaults | [Split Documents](chunking.md) — canonical subsection: [Token-based splitting and tokenizers](chunking.md#token-based-splitting-and-tokenizers) |
| Pipeline NIMs and hardware | [Support Matrix](support-matrix.md) |
| Metadata / job schema fields | [Metadata Reference](content-metadata.md) |
| `vdb_upload` and `dense_dim` | [Data Store](data-store.md) |
| Known issues, deprecations, NIM caveats | [Release Notes](releasenotes-nv-ingest.md) |

### Pattern for any architecture or code change

For **any** future change in architecture or code that affects the docs (not just tokenizers), follow this pattern:

1. **Identify the concept**  
   Decide what concept changed (e.g. tokenizer, embedding model, environment variable, microservice, job schema field, API parameter).

2. **Choose the canonical home**  
   Decide which doc section or page should be the single canonical home for this concept (e.g. [Environment Variables](environment-config.md), Architecture Overview, [Token-based splitting and tokenizers](chunking.md#token-based-splitting-and-tokenizers), job configuration schema). If no suitable place exists, create a new subsection and make it discoverable from the main index or [overview](overview.md).

3. **Update the canonical description**  
   Update that canonical section first with: current behavior; defaults and configuration options; external requirements (e.g. NGC or Hugging Face account, licensing, tokens).

4. **De-duplicate and re-point**  
   Search the repo for all mentions of that concept (names, env vars, model IDs, API fields). For each mention: replace copied detailed explanations with concise text that defers to the canonical section; ensure any remaining text is fully consistent with the canonical description.

**Example — tokenizer changes:** Canonical home is [Token-based splitting and tokenizers](chunking.md#token-based-splitting-and-tokenizers). After updating it, search for: `llama-tokenizer`, `Llama tokenizer`, `meta-llama`, `DOWNLOAD_LLAMA_TOKENIZER`, `HF_ACCESS_TOKEN`, `token-based splitting`, and any model names that were previously the default; then shorten or link each match to the canonical section.

### Guardrails

- **Never** introduce a new detailed explanation of a previously documented concept without either:
  - Moving that explanation into the concept’s canonical section, or
  - Explicitly updating the canonical section and linking back to it from the new text.
- **Avoid** having two places where a reader could reasonably believe they are both “the main description” of the same behavior. If in doubt, keep one canonical description and have the other place summarize in 1–2 sentences and link.

### Style and linking conventions

When pointing to a canonical section:

- **Use consistent phrasing**, for example:
  - “For the most up-to-date tokenizer configuration and requirements, see [Token-based splitting and tokenizers](chunking.md#token-based-splitting-and-tokenizers).”
  - “For full details, see the [Environment Variables](environment-config.md) reference.”
  - “For full details and the latest [concept] behavior, see [canonical section](link).” (Replace [concept] with the topic, e.g. tokenizer, environment variables.)
- **Keep cross-references stable:**
  - Prefer relative links and section headings that are unlikely to change (e.g. `chunking.md#token-based-splitting-and-tokenizers`).
  - If you rename or move a canonical section, update all inbound links in the same commit.
