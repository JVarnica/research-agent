# research-agent

A containerised deep research agent built with **LangGraph**. Takes a question, runs iterative search → scrape → summarise → reflect loops, and writes a cited report. Frontend-agnostic — emits progress events that any client can consume.

Built for [ExecuChat](https://github.com/JVarnica/Execu_Chat) but deployable standalone.

> 📖 **Full write-up — graph design, claims model, and reflection loop:** [Research Agent hub](https://jvarnica.github.io/projects/research-agent/)


## How It Works

```
Query → Plan queries → [Search → Scrape → Summarise → Extract claims] → Reflect → Loop? → Write sections → Stitch report
                              ↑__________________________|  (if evidence insufficient)

A FastAPI endpoint queues a task; a worker runs it through the LangGraph graph, emitting events at each phase (`planning`, `searching`, `doc_summarised`, `claims_extracted`, `section_written`, `complete`). The client polls for events and renders progress.

**Design highlights:**
- **Claims as the unit of knowledge** — atomic, source-cited claims are extracted before writing, so report sections stay grounded and citations are auditable.
- **Reflection with gap detection** — the reflect node identifies the *specific* missing evidence and generates targeted follow-up queries, marking unfillable gaps rather than looping forever.
- **Parallel-safe state** — a typed `OverallState` with custom reducers accumulates queries, documents, summaries, and claims across parallel branches without race conditions.
- **Redis checkpointing** — graph state persists via LangGraph's `AsyncRedisSaver`, so long-running tasks survive restarts.

## Stack

LangGraph · FastAPI · Redis · SearxNG (search) · vLLM (LLM calls) · Docker


## Running

This container is designed to run as part of the [vllm-server](https://github.com/JVarnica/vllm-server) Compose stack, which provides its dependencies (vLLM, SearxNG, Redis). It expects:

```bash
SEARXNG_URL=...        # search backend
VLLM_URL=...           # inference endpoint
VLLM_MODEL=...         # served model
REDIS_URL=...          # state + queue
MAX_CONCURRENT_TASKS=2
```

To run inside the full stack:

```bash
docker compose up -d deep-research
```

## Related

- [ExecuChat](https://github.com/JVarnica/Execu_Chat) — Android frontend
- [vllm-server](https://github.com/JVarnica/vllm-server) — backend stack
- Write-ups: [making a research agent](https://jvarnica.github.io/Research-agent/) · [incrementally building research agents](https://jvarnica.github.io/Building-research-agent/)
