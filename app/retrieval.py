import logging
import uuid
import json
from langgraph.types import Send
from pydantic import BaseModel, Field

from .task import _emit, today_str
from .llm import get_clients
from .state import OverallState, Document, QuerySetLLM, Query
from .search import search_query, scrape_hits
from .prompts import QUERY_GEN_PROMPT


logger = logging.getLogger(__name__)


async def generate_queries(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="generating_queries", message="Planning search strategy")
    
    structured = get_clients().struct_cheap_llm(QuerySetLLM)
    result: QuerySetLLM = await structured.ainvoke([
        {"role": "system", "content": QUERY_GEN_PROMPT.format(current_date=today_str())},
        {"role": "user", "content": state["original_query"]},
    ])
    
    # Model no need to make ids anymore
    queries = [
        Query(id=f"q_{uuid.uuid4().hex[:8]}", 
              **query.model_dump())
        for query in result.queries
    ]
    _emit(task_id, "queries_generated", count=len(queries),
          queries=[{"query": q.query, "rationale": q.rationale, "category": q.category} for q in queries]) #emits all queries 
    return {"search_queries": queries}

# ============================================================
# Node 2: SEquential Search 
# ============================================================

async def search_queries_seq(state: OverallState) -> dict:
    task_id = state["task_id"]

    already = set(state.get("searched_queries_ids", []))
    pending = [q for q in state["search_queries"] if q.id not in already]

    seen_urls: set[str] = set(state.get("seen_urls", []))
    new_hits: list[Document] = []
    completed_queries_id: list[str] = []

    for query in pending:
        _emit(task_id, "searching", query=query.query)
        hits = await search_query(query, seen_urls=seen_urls, max_results=15)
        completed_queries_id.append(query.id)

        for h in hits:
            if h.url not in seen_urls:
                new_hits.append(h)
                seen_urls.add(h.url)
        _emit(task_id, "search_complete",
              query=query.query, hit_count=len(hits))

    return {
        "search_hits": new_hits,
        "searched_queries_ids": completed_queries_id,
        "seen_urls": [h.url for h in new_hits],
    }


async def scrape_node(state: OverallState) -> dict:
    task_id = state["task_id"]
    hits = state.get("search_hits", [])
    if not hits:
        _emit(task_id, "scrape_complete", scraped=0)
        return {"raw_docs": []}

    _emit(task_id, "stage", stage="scraping",
          message=f"Scraping {len(hits)} approved URLs")
    docs = await scrape_hits(hits)
    _emit(task_id, "scrape_complete", scraped=len(docs))
    return {"raw_docs": docs}
