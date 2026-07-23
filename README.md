# research-agent

A containerised deep research agent built with **LangGraph**. Takes a question, runs iterative search → scrape → summarise → reflect loops, and writes a cited report. Frontend-agnostic — emits progress events that any client can consume.

Built for [ExecuChat](https://github.com/JVarnica/Execu_Chat) but deployable standalone.

> 📖 **Full write-up — graph design, claims model, and reflection loop:** [Research Agent hub](https://jvarnica.github.io/projects/research-agent/)


## How It Works

A FastAPI endpoint queues a task; a worker runs it through the LangGraph graph, emitting events at each phase (`planning`, `searching`, `doc_summarised`, `claims_extracted`, `section_written`, `complete`). The client polls for events and renders progress.

<img width="780" height="984" alt="architecture-diagram-1x" src="https://github.com/user-attachments/assets/7941dfe8-badb-440f-a47f-41d9f657b23b" />


### Key Design Decisions

**Claims as topic map** — Aggregates the doc summaries into topic-level claims, it's a topic label with the doc_ids. This maps documents to specific topics making it easier for planning, as will just put the documents from that specific topic for different sections.

**Section Writer sees summaries & claims** - The doc summaries have the facts and quotes, claims just more info on topic. By having the information not aggregated it can write proper grounded reports not generalizations. 

**Reflection loop with gap detection** — the reflect node doesn't just ask "is this enough?" — it identifies the specific knowledge gap and generates targeted follow-up queries. If the same gap remains unfilled after a search loop, the node marks it as unfillable and proceeds rather than looping indefinitely.

**Reflection current understanding** - the reflect node doesn't just have a gap, it now also has current understanding so it knows what it has learned so far. So it can build on what it knows, this was added as would say the same gap in further loops now it has stopped.

**Worker queue over background tasks** — tasks are pushed to a Redis list and processed by a dedicated worker loop. This is more robust than asyncio background tasks for long-running jobs and survives server restarts without losing queued work.

**Frontend-agnostic event stream** — the agent emits named events at each phase (`status`, `planning`, `searching`, `doc_summarised`, `claims_extracted`, `section_written`, `complete`). These events are added on a Redis List, so the frontend can then poll this list for information.

**Redis checkpointing** — graph state persists via LangGraph's `AsyncRedisSaver`, so long-running tasks survive restarts.


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
