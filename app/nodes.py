import asyncio
import json
import logging
import uuid
import re
from typing import Any
import httpx
from pydantic import BaseModel, Field

from langgraph.types import Send
from .state import (
    HitVerdict, OverallState, Query, Document, DocSummary, Claim, SearchHit,
    ReportPlan, SectionPlan, WrittenSection, ReflectionResult,
)
from .llm import get_clients
from .search import search_query, _collect_ordered_doc_ids, _rewrite_citations, scrape_hits
from .events import get_channel

logger = logging.getLogger(__name__)


def _emit(task_id: str, event: str, **data: Any):
    """Fire-and-forget event emission to Redis ."""
    ch = get_channel(task_id)
    asyncio.create_task(ch.emit(event, data)) # coroutine creation on events


# ============================================================
# Node 1: generate initial queries from the original question
# ============================================================

QUERY_GEN_PROMPT = """You are a research planner. Generate 3-5 diverse search queries
to investigate the user's question. Each query explores a different angle and has
a one-sentence rationale.

Route each query to one category:
- Technical / ML / scientific questions: mix 'science' (papers, theory) with
  'it' (libraries, docs, implementations), and optionally 'general' for context.
- Historical / cultural questions: mix 'general' (context, narrative) with
  'science' (academic analysis). Do NOT use 'it' for non-technical topics.
- Current events: include 'news' alongside 'general'.

Default to 'general' when unsure."""


class InitialQueries(Query.__class__):  # placeholder, see below
    pass

class QuerySet(BaseModel):
    """A diverse set of queries covering different angles of the question."""
    queries: list[Query] = Field(
        min_length=2, max_length=5,
        description=(
            "3-5 queries, each exploring a distinct angle. "
            "No near-duplicates — two queries that differ only in wording count as one."
        ),
    )

async def generate_queries(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="generating_queries", message="Planning search strategy")
    
    structured = get_clients().struct_cheap_llm(QuerySet)
    result: QuerySet = await structured.ainvoke([
        {"role": "system", "content": QUERY_GEN_PROMPT},
        {"role": "user", "content": state["original_query"]},
    ])
    
    # Stamp fresh IDs (model may or may not provide them; we own ID space)
    queries = [
        Query(id=f"q_{uuid.uuid4().hex[:8]}", query=q.query, 
              rationale=q.rationale, category=q.category)
        for q in result.queries
    ]
    _emit(task_id, "queries_generated", count=len(queries),
          queries=[{"query": q.query, "rationale": q.rationale, "category": q.category} for q in queries]) #emits all queries 
    return {"search_queries": queries}

# ============================================================
# Node 2: SEquential Search 
# ============================================================

async def search_queries_seq(state: OverallState) -> dict:
    task_id = state["task_id"]
    already = {h.source_query_id for h in state.get("search_hits", [])}
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
    hits = state.get("search_hits", []) #search hits per loop overwrite so will only get new hits
    if not hits:
        return {"raw_docs": []}

    _emit(task_id, "stage", stage="scraping",
          message=f"Scraping {len(hits)} approved URLs")
    docs = await scrape_hits(hits)
    _emit(task_id, "scrape_complete", scraped=len(docs))
    return {"raw_docs": docs}

# ============================================================
# Node: fan-out summarization — one branch per doc
# ============================================================

def fan_out_summarize(state: OverallState) -> list[Send]:
    summarized_ids = {s.doc_id for s in state.get("doc_summaries", [])}
    pending = [d for d in state["raw_docs"] if d.id not in summarized_ids]
    if not pending:
        return "extract_claims"
    
    return [
        Send("summarize_one_doc", {
            "task_id": state["task_id"],
            "doc": d,
            "original_query": state["original_query"],
        })
        for d in pending
    ]

SUMMARIZE_PROMPT = """You are reading ONE source document to extract findings relevant to a research question.

Research question: {question}

First, relevance:
- Set relevant=false if the document does not actually address the question. A document that only shares a word with the question in an unrelated context is NOT relevant.
- If relevant=false, return an empty key_findings and quotes list.

Rules:
- Set relevant=false if the document does not actually address the question.
- key_findings: 3-8 findings. DO NOT invent findings to reach count. 2 sharp findings are better than 3 padded ones
- Each finding must state a specific date, number, named person/place/event or causal mechanism. 
- A causal-mechanism finding states cause then effect (X caused Y because Z), not a vague description.
- Do not hedge or generalize. Quote specifics.
- quotes: up to 4 short verbatim quotes (under 25 words each) that support your findings. Omit if add nothing

Document title: {title}
URL: {url}

Document content:
{content}"""

async def summarize_one_doc(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    doc: Document = branch_input["doc"]
    
    structured = get_clients().struct_cheap_llm(DocSummary)
    try:
        summary: DocSummary = await structured.ainvoke([
            {"role": "user", "content": SUMMARIZE_PROMPT.format(
                question=branch_input["original_query"],
                title=doc.title,
                url=doc.url,
                content=doc.raw_content,
            )},
        ])
        # The schema lets the model leave doc_id/url unset; we own them
        summary.doc_id = doc.id
        summary.url = doc.url
        summary.title = doc.title
        logger.debug("summary doc_id=%s url=%s relevant=%s findings=%s quotes=%s",
                doc.id, doc.url, summary.relevant,
                summary.key_findings, summary.quotes)
    except Exception as e:
        logger.exception(f"summarize failed for {doc.url}: {e}")
        # Return a "not relevant" placeholder so the fan-out completes cleanly
        summary = DocSummary(doc_id=doc.id, url=doc.url, relevant=False, 
                             key_findings=[], quotes=[])
    
    _emit(task_id, "doc_summarized", doc_title=summary.title, key_findings=summary.key_findings, relevant=summary.relevant)
    return {"doc_summaries": [summary]}

# ============================================================
# Node 4: extract claims from all relevant summaries
# ============================================================

EXTRACT_PROMPT = """You are aggregating findings from multiple sources into ATOMIC factual claims.

Research question: {question}

You will receive a JSON list of document summaries, each with a doc_id and key_findings
-  One claim = ONE idea, stated WITH its specifics: include the relevant date, number, named entity, or causal mechanism.
- No adjectives, framing, or filler. If a sentence contains no specific, it is not a claim — drop it.
- Cite source documents by their doc_id in source_doc_ids.
- Multiple sources supporting the same claim → high confidence; single source → medium; conflicting → low.
- Do NOT include generic background or filler. Only specific claims that help answer the question.

Summaries:
{summaries_json}"""


class ClaimSet(BaseModel):
    claims: list[Claim] = Field(min_length=1, max_length=40)


async def extract_claims(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="extracting_claims", message="Building claims from sources")

    covered_doc_ids = {did for c in state.get("claims", []) for did in c.source_doc_ids}
    new_relevant = [s for s in state["doc_summaries"] if s.relevant and s.doc_id not in covered_doc_ids]
    if not new_relevant:
        _emit(state["task_id"], "claims_extracted", count=0)
        logger.warning(f"task {task_id}: no relevant summaries to extract claims from")
        return {"claims": []}
    
    summaries_json = json.dumps([s.model_dump() for s in new_relevant])
    structured = get_clients().structured_llm(ClaimSet)
    
    result: ClaimSet = await structured.ainvoke([
        {"role": "user", "content": EXTRACT_PROMPT.format(
            question=state["original_query"],
            summaries_json=summaries_json,
        )},
    ])
    # Own the IDs
    claims = [
        Claim(id=f"c_{uuid.uuid4().hex[:8]}", statement=c.statement,
              source_doc_ids=c.source_doc_ids, confidence=c.confidence)
        for c in result.claims
    ]
    logger.debug("claims extracted: %s", claims)
    _emit(task_id, "claims_extracted", count=len(claims))
    return {"claims": claims}


# ============================================================
# Node 5: reflect — do we need another search loop?
# ============================================================

REFLECT_PROMPT = """You are auditing research progress. Decide if the gathered claims sufficiently \
answer the research question, or if a follow-up search loop is needed.

Research question: {question}

Previous understanding (from prior reflection):
{previous_understanding}

Current claims:
{claims_json}

Queries already searched ( do not repeat or paraphrase these):
{previous_queries}

Has this new evidence changed your understanding? Are these claims sufficient or is there still a gap?

If sufficient, set is_sufficient=true and leave follow_up_queries empty.

If gaps exist, set is_sufficient=false, describe the gap in one sentence, and provide 1-3 \
follow-up queries targeting that gap.Each follow-up must explore a distinctly \
different angle from what's already been tried — different keywords, different framing, \
or a different sub-question. If you can't think of genuinely new angles, set \
is_sufficient=true instead of repeating."""


async def reflect(state: OverallState) -> dict:
    task_id = state["task_id"]
    loop = state.get("research_loop_count", 0)
    _emit(task_id, "stage", stage="reflecting", loop=loop)
    
    claims_json = json.dumps([c.model_dump() for c in state["claims"]])
    prior_queries = state.get("search_queries", [])
    previous_queries = ( "\n".join(f"- {q.query}" for q in prior_queries) or "(none yet)")

    structured = get_clients().structured_llm(ReflectionResult)
    
    result: ReflectionResult = await structured.ainvoke([
        {"role": "user", "content": REFLECT_PROMPT.format(
            question=state["original_query"], claims_json=claims_json, previous_queries=previous_queries, previous_understanding=state.get("understanding_history", "")
        )},
    ])
    
    # Stamp IDs on follow-ups
    new_queries = [
        Query(id=f"q_{uuid.uuid4().hex[:8]}", query=q.query, rationale=q.rationale, category=q.category)
        for q in result.follow_up_queries
    ]
    is_sufficient = result.is_sufficient or not new_queries
    if not is_sufficient and not new_queries:
        logger.warning("reflect: all follow-ups were duplicates — forcing termination")

    _emit(task_id, "reflection", 
          sufficient=is_sufficient, 
          gap=result.knowledge_gap if not is_sufficient else "",
          new_queries=len(new_queries))
    
    return {
        "is_sufficient": is_sufficient,
        "search_queries": new_queries,  # extends via operator.add
        "research_loop_count": loop + 1,
        "understanding_history": [result.current_understanding],  # extends via operator.add
    }


def reflect_router(state: OverallState) -> str:
    """Decide: another search loop, or move to planning?"""
    if state["is_sufficient"]:
        return "plan_report"
    if state["research_loop_count"] >= state["max_research_loops"]:
        return "plan_report"
    return "search_all_queries"


# ============================================================
# Node 6: plan the report structure
# ============================================================

PLAN_PROMPT = """Plan a research report answering: {question}

You have these claims to organize:
{claims_json}

Produce a plan with 3-5 sections. Each section:
- Has a distinct angle (no overlap between sections).
- References specific claim IDs that belong in it.
- Every claim should belong to one section ONLY if relevant; orphan claims OK if minor.

Section ordering should flow logically (background → core → implications, or by theme)."""


async def plan_report(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="planning", message="Structuring the report")
    
    claims_json = json.dumps([c.model_dump() for c in state["claims"]])
    structured = get_clients().structured_llm(ReportPlan)
    
    plan: ReportPlan = await structured.ainvoke([
        {"role": "user", "content": PLAN_PROMPT.format(
            question=state["original_query"], claims_json=claims_json,
        )},
    ])
    # Stamp section IDs
    plan = ReportPlan(
        title=plan.title,
        sections=[
            SectionPlan(id=f"s_{i}", title=s.title, angle=s.angle, claim_ids=s.claim_ids)
            for i, s in enumerate(plan.sections)
        ],
    )
    _emit(task_id, "plan_ready", 
          title=plan.title, 
          sections=[{"title": s.title, "claim_count": len(s.claim_ids)} for s in plan.sections])
    return {"plan": plan}


# ============================================================
# Node 7: fan-out section writing
# ============================================================

def fan_out_sections(state: OverallState) -> list[Send]:
    claims_by_id = {c.id: c for c in state["claims"]}
    docs_by_id = {d.id: d for d in state["raw_docs"]}
    
    sends = []
    for section in state["plan"].sections:
        section_claims = [claims_by_id[cid] for cid in section.claim_ids if cid in claims_by_id]
        # Resolve doc URLs for citation
        doc_refs = {}
        for c in section_claims:
            for did in c.source_doc_ids:
                if did in docs_by_id and did not in doc_refs:
                    doc_refs[did] = docs_by_id[did].url
        
        sends.append(Send("write_section", {
            "task_id": state["task_id"],
            "section": section,
            "claims": section_claims,
            "doc_refs": doc_refs,
            "original_query": state["original_query"],
        }))
    return sends


WRITE_SECTION_PROMPT = """You are writing ONE section of a research report.

DO NOT attempt to include section title or any heading in the output — just write the body. 
The stitcher will handle formatting. 

Research question: {question}
Section title: {title}
Section angle: {angle}

Use ONLY these claims. Cite by claim ID in brackets, e.g. [c_a1b2c3].

Claims:
{claims_json}

Write 200- 800 words of flowing prose for a curious reader.
-Connect facts with reasoning words: 'because', 'therefore', 'however', 
'consequently', etc. Don't just list facts in a sequence.
-Be specific about names, dates and figures. Avoid generalizations and hedging.
-Use only facts present in the provided claims. Do not add names, dates, or details not in the claims, even if you know them."
-Do not end the section with a summary paragraph; stop when the content is covered.

Output markdown."""


async def write_section(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    section: SectionPlan = branch_input["section"]
    claims = branch_input["claims"]
    
    _emit(task_id, "writing_section", title=section.title)
    
    claims_json = json.dumps([c.model_dump() for c in claims])
    llm = get_clients().writer_llm(temperature=0.5, max_tokens=4096)
    
    resp = await llm.ainvoke([
        {"role": "user", "content": WRITE_SECTION_PROMPT.format(
            question=branch_input["original_query"],
            title=section.title,
            angle=section.angle,
            claims_json=claims_json,
        )},
    ])
    body = resp.content
    
    # Strip Qwen3 <think> tags if vLLM didn't already (reasoning_parser should handle it)
    import re
    body = re.sub(r"<think>.*?</think>", "", body, flags=re.DOTALL).strip()
    
    # Find which claim IDs were actually cited
    cited = re.findall(r"\[c_[a-f0-9]+\]", body)
    citations_used = list(set(c.strip("[]") for c in cited))
    
    written = WrittenSection(
        id=section.id,
        title=section.title,
        body_markdown=body,
        citations_used=citations_used,
    )
    _emit(task_id, "section_written", title=section.title, citations=len(citations_used))
    return {"written_sections": [written]}


# ============================================================
# Node 8: stitch sections into the final report
# ============================================================

async def stitch_report(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="stitching", message="Assembling final report")
    
    # Sections come back from fan-out in arbitrary order; restore plan order
    sections_by_id = {s.id: s for s in state["written_sections"]}
    ordered = [sections_by_id[sp.id] for sp in state["plan"].sections if sp.id in sections_by_id]
    
    # Build the references list from all citations used across sections
    docs_by_id = {d.id: d for d in state["raw_docs"]}
    claims_by_id = {c.id: c for c in state["claims"]}
    
    # Walk the actual citation markers, not citations_used metadata — markers are truth
    ordered_doc_ids = _collect_ordered_doc_ids(ordered, claims_by_id)
    doc_to_ref = {did: i + 1 for i, did in enumerate(ordered_doc_ids)}

    references = []
    for did in ordered_doc_ids:
        d = docs_by_id.get(did)
        if d is not None:
            references.append(f"{doc_to_ref[did]}. [{d.title or d.url}]({d.url})")

    
    # Assemble — no LLM call here, deterministic stitch. We get the model 
    # to write a short intro/conclusion in a single small call.
    intro_conclusion = await _write_intro_conclusion(state, ordered)
    
    report_parts = [
        f"# {state['plan'].title}",
        "",
        intro_conclusion["intro"],
        "",
    ]
    for sec in ordered:
        rewritten = _rewrite_citations(sec.body_markdown, claims_by_id, doc_to_ref)
        report_parts.extend([f"## {sec.title}", "", rewritten, ""])
    
    report_parts.extend([
        "## Conclusion", "",
        intro_conclusion["conclusion"], "",
        "## References", "",
        *references,
    ])
    
    final = "\n".join(report_parts)
    _emit(task_id, "report_ready", length=len(final))
    return {"final_report": final}


class IntroConclusion(BaseModel):
    intro: str = Field(
        description="Few sentence opening that frames the question",
        max_length=900
        )
    conclusion: str = Field(
        description="3-4 sentence conclusion summarizing the answer",
        max_length=800)


async def _write_intro_conclusion(state: OverallState, sections: list[WrittenSection]) -> dict:
    section_summaries = "\n".join(f"- {s.title}: {s.body_markdown[:200]}..." for s in sections)
    structured = get_clients().struct_cheap_llm(IntroConclusion)
    result: IntroConclusion = await structured.ainvoke([
        {"role": "user", "content": (
            f"Write an intro and conclusion for a report answering: {state['original_query']}\n\n"
            f"The report has these sections:\n{section_summaries}\n\n"
            f"Intro frames the question (2-3 sentences). Conclusion summarizes the answer (2-3 sentences). "
            f"Do not introduce new facts."
        )},
    ])
    return {"intro": result.intro, "conclusion": result.conclusion}