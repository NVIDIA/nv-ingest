# Agentic retrieval (concept)

**Agentic retrieval** is iterative, tool-driven retrieval. A large language
model (LLM) agent plans steps and issues search tool calls until it has enough
context. In `select` mode, the workflow fuses candidates and ranks documents.
In `answer` mode, the agent instead produces an integrated answer with citations
validated against documents retrieved during that run. **One-pass retrieval**
sends a single static query through dense or hybrid search and returns
chunk-level hits.

NeMo Retriever Library includes first-class agentic selection and answer paths.
`retriever query --agentic`, `POST /v1/query` with `agentic=true`, and the
`agentic_query` Model Context Protocol (MCP) tool run selection mode.
`retriever query --agentic --agentic-mode answer`, `/v1/query` with
`agentic_mode=answer`, `/v1/answer` with `mode=agentic`, the Python service
client's `agentic_answer` methods, and the MCP `agentic_answer` tool run answer
mode. Both use a Reason and Act (ReAct) loop over the same LanceDB table that
one-pass retrieval uses. You do not have to implement the agent loop in
application code.

Selection mode ranks documents rather than chunks. CLI `--agentic` output is
the retrieval-hop hit plus `doc_id`, `rank`, and `result_source`; it is not the
five-field dense CLI projection. Answer mode terminates inside the ReAct loop
and bypasses reciprocal rank fusion and final document selection. It returns an
answer, citation IDs, hydrated citation hits, and status/error metadata.

Local CLI runs default to an in-process vLLM agent LLM. Retriever
Service requires a remote OpenAI-compatible chat-completions endpoint. A
self-hosted vLLM-backed NIM must enable automatic tool choice and a tool-call
parser. Helm `answer_llm` configures the classic `/v1/answer` path; agentic
answer mode additionally requires `serviceConfig.agentic`.

For commands, service configuration, request and response contracts, and failure behavior, refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

## Related Topics { #related-topics }

- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Starter kits](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/README.md)
