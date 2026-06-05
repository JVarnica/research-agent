import asyncio
import json
import logging
from time import time
import uuid
import re
from typing import Any, Literal
import httpx
from pydantic import BaseModel, Field

from langgraph.types import Send
from .state import (
    HitVerdict, OverallState, Query, Document, DocSummary, Claim, SearchHit,
    ReportPlan, SectionPlan, WrittenSection, ReflectionResult, ExtractedClaim 
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

    new_hits: list[SearchHit] = []
    for query in pending:
        _emit(task_id, "searching", query=query.query)
        hits = await search_query(query, seen_urls=seen_urls, max_results=15)
        for h in hits:
            if h.url not in seen_urls:
                new_hits.append(h)
                seen_urls.add(h.url)
        _emit(task_id, "search_complete",
              query=query.query, hit_count=len(hits))

    return {
        "search_hits": new_hits,
        "seen_urls": [h.url for h in new_hits],
    }

TRIAGE_CHUNK = 12

async def _triage_chunk(question: str, chunk: list[SearchHit]) -> set[str]:
    t0 = time.time()
    hits_json = json.dumps([
        {"id": h.id, "title": h.title, "url": h.url, "snippet": h.snippet}
        for h in chunk
    ])
    structured = get_clients().struct_cheap_llm(ScrapeDecision)
    try:
        decision: ScrapeDecision = await structured.ainvoke([
            {"role": "user", "content": PRE_SCRAPE_PROMPT.format(question=question, hits_json=hits_json)},
        ])
        valid = {h.id for h in chunk}
        logger.debug(f"_triage_chunk: triaged {len(chunk)} hits in {round(time.time() - t0, 1)}s, keeping {len(decision.verdicts)}")
        return {v.id for v in decision.verdicts if v.keep and v.id in valid}
    except Exception as e:
        logger.warning(f"triage chunk failed, keeping none: {e}")
        return set()

class ScrapeDecision(BaseModel):
    verdicts: list[HitVerdict] = Field(min_length=1, max_length=50)
#========================================================
#Pre-scrape sort
#========================================================
PRE_SCRAPE_PROMPT = """You are triaging search results before scraping. Scraping is expensive — be STRICT.

Research question: {question}

You will receive search hits (id, title, URL, snippet). Judge EVERY hit individually from its title and snippet alone and return a verdict for each.

Default to keep=false. Set keep=true ONLY if the title or snippet names something SPECIFIC to the research question — \
a relevant person, place, event, date, or clearly on-topic discussion. A hit that merely shares a common word with the \
question ("role", "fall", "cause", "long", "key") in an unrelated context is NOT relevant.

Always drop (keep=false):
- Dictionary, thesaurus, grammar, or vocabulary pages ("Definition of…", "100 Words to Use Instead of…", Wiktionary).
- Software/technical documentation unrelated to the question (Azure roles, SQL Server, API docs) — unless the question is about that software.
- Medical/health pages unless the question is medical.
- OS or product help pages ("How to get help in Windows").
- Generic hub/category pages, vendor landing pages, SEO listicles.
- Video, forum, or Q&A links unless the snippet clearly shows substantive on-topic content.

When unsure, drop it. There are plenty of hits; a wasted scrape costs more than a missed marginal page.

Hits:
{hits_json}
"""

async def pre_scrape(state: OverallState) -> dict:
    task_id = state["task_id"]
    hits: list[SearchHit] = state.get("search_hits", [])
    if not hits:
        return {"hits_to_scrape": []}

    _emit(task_id, "stage", stage="triaging",
          message=f"Triaging {len(hits)} hits before scrape")

    MAX_SCRAPE = 15  # ceiling guard against pathological over-keeping
    try:
        chunks = [hits[i:i + TRIAGE_CHUNK] for i in range(0, len(hits), TRIAGE_CHUNK)]
        results = await asyncio.gather(*[
            _triage_chunk(state["original_query"], chunk) for chunk in chunks
        ])
        keep_ids = set().union(*results) if results else set()
        kept = [h for h in hits if h.id in keep_ids]

        MAX_SCRAPE = 15
        if len(kept) > MAX_SCRAPE:
            kept = sorted(kept, key=lambda h: h.search_score, reverse=True)[:MAX_SCRAPE]

        logger.info(f"pre_scrape: {len(hits)} hits → {len(kept)} kept across {len(chunks)} chunks")
        _emit(task_id, "triage_complete", before=len(hits), after=len(kept))
      
        if len(kept) > MAX_SCRAPE:
            kept = sorted(kept, key=lambda h: h.search_score, reverse=True)[:MAX_SCRAPE]
    except Exception as e:
        logger.exception(f"pre_scrape failed, falling back to top-N by score: {e}")
        kept = sorted(hits, key=lambda h: h.search_score, reverse=True)[:MAX_SCRAPE]

    logger.info(f"pre_scrape: {len(hits)} hits → {len(kept)} kept")
    _emit(task_id, "triage_complete", before=len(hits), after=len(kept))
    return {"hits_to_scrape": kept}

async def scrape_node(state: OverallState) -> dict:
    task_id = state["task_id"]
    hits = state.get("hits_to_scrape", [])
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
- If relevant=false, return an empty overview, key_findings and quotes list.

Rules:
- overview: 1-2 sentences orienting a downstream writer on what THIS document is — its angle, scope, or argument.
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

Existing claims already extracted from earlier loops:
{existing_claims_json}

Summaries:
{summaries_json}

You will receive a JSON list of document summaries, each with a doc_id and key_findings
-  One claim = ONE idea, stated WITH its specifics: include the relevant date, number, named entity, or causal mechanism.
- No adjectives, framing, or filler. If a sentence contains no specific, it is not a claim — drop it.
- Cite source documents by their doc_id in source_doc_ids.
- Multiple sources supporting the same claim → high confidence; single source → medium; conflicting → low.
- Do NOT include generic background or filler. Only specific claims that help answer the question.


"""
 

class ClaimSet(BaseModel):
    claims: list[ExtractedClaim] = Field(min_length=1, max_length=40)


async def extract_claims(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="extracting_claims", message="Aggregating sources into topic claims")
 
    existing_claims = state.get("claims", [])
    covered_doc_ids = {did for c in existing_claims for did in c.source_doc_ids}
    new_relevant = [
        s for s in state["doc_summaries"]
        if s.relevant and s.doc_id not in covered_doc_ids
    ]
    if not new_relevant:
        _emit(task_id, "claims_extracted", count=0, merged=0)
        logger.warning(f"task {task_id}: no relevant summaries to extract claims from")
        return {"claims": []}
 
    existing_compact = [
        {"id": c.id, "statement": c.statement, "source_count": len(c.source_doc_ids)}
        for c in existing_claims
    ]
    existing_claims_json = json.dumps(existing_compact, ensure_ascii=False)
    summaries_json = json.dumps([s.model_dump() for s in new_relevant], ensure_ascii=False)
 
    structured = get_clients().structured_llm(ClaimSet)
    result: ClaimSet = await structured.ainvoke([
        {"role": "user", "content": EXTRACT_PROMPT.format(
            question=state["original_query"],
            existing_claims_json=existing_claims_json,
            summaries_json=summaries_json,
        )},
    ])
    existing_by_id = {c.id: c for c in existing_claims}
    out: list[Claim] = []
    new_count = 0
    merged_count = 0
    for ec in result.claims:
        if ec.merges_into and ec.merges_into in existing_by_id:
            # Re-emit with the SAME id — the merge_claims_by_id reducer will replace
            # the existing entry with this merged version.
            prev = existing_by_id[ec.merges_into]
            combined = list(dict.fromkeys([*prev.source_doc_ids, *ec.source_doc_ids]))
            out.append(Claim(
                id=prev.id,
                statement=prev.statement,  # keep canonical wording
                source_doc_ids=combined,
                confidence=ec.confidence,
            ))
            merged_count += 1
        else:
            out.append(Claim(
                id=f"c_{uuid.uuid4().hex[:8]}",
                statement=ec.statement,
                source_doc_ids=ec.source_doc_ids,
                confidence=ec.confidence,
            ))
            new_count += 1
 
    logger.debug("claims: %d new, %d merged", new_count, merged_count)
    _emit(task_id, "claims_extracted", count=new_count, merged=merged_count)

    return {"claims": out}

# ============================================================
# Node 5: reflect — do we need another search loop?
# ============================================================

REFLECT_PROMPT = """You are auditing research progress. Decide if the gathered claims sufficiently \
answer the research question, or if a follow-up search loop is needed.

Research question: {question}

Previous understanding (from prior reflection):
{previous_understanding}

Topics covered so far (claim labels — quick index of what's been touched):
{claims_json}
 
Evidence gathered (the actual content — overview + key_findings per document):
{evidence_json}
 
Queries already searched (do not repeat or paraphrase these):
{previous_queries}
 
When judging sufficiency, look at the EVIDENCE not just the topic index. A topic can be \
'covered' as a label while the underlying evidence is thin, one-sided, or all from the same \
source type (e.g. only blog posts, no primary sources). Flag those as gaps.


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
    
    claims_json = json.dumps(
        [{"id": c.id, "topic": c.statement, "sources": len(c.source_doc_ids),
          "confidence": c.confidence} for c in state["claims"]],
        ensure_ascii=False,
    )
    evidence_json = json.dumps(
        [{"doc_id": s.doc_id, "title": s.title,
          "overview": s.overview, "key_findings": s.key_findings}
         for s in state["doc_summaries"] if s.relevant],
        ensure_ascii=False,
    )

    prior_queries = state.get("search_queries", [])
    previous_queries = ( "\n".join(f"- {q.query}" for q in prior_queries) or "(none yet)")

    structured = get_clients().structured_llm(ReflectionResult)
    
    result: ReflectionResult = await structured.ainvoke([
        {"role": "user", "content": REFLECT_PROMPT.format(
            question=state["original_query"], claims_json=claims_json, 
            evidence_json=evidence_json, previous_queries=previous_queries, previous_understanding=state.get("understanding_history", "")
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

MAX_SUMMARIES_P_SEC = 7

def fan_out_sections(state: OverallState) -> list[Send]:
    claims_by_id = {c.id: c for c in state["claims"]}
    summaries_by_id = {
        s.doc_id: s for s in state["doc_summaries"] if s.relevant
    }
 
    sends = []
    for section in state["plan"].sections:
        section_claims = [
            claims_by_id[cid] for cid in section.claim_ids if cid in claims_by_id
        ]
 
        # Score each supporting doc by how many of THIS section's claims it backs.
        # A doc covering 4 of the section's 5 claims is more central evidence than
        # one covering only 1. Tiebreak by summary richness (findings + quotes).
        support_count: dict[str, int] = {}
        for claim in section_claims:
            for doc_id in claim.source_doc_ids:
                if doc_id in summaries_by_id:
                    support_count[doc_id] = support_count.get(doc_id, 0) + 1
 
        ranked_doc_ids = sorted(
            support_count.keys(),
            key=lambda did: (
                -support_count[did],
                -(len(summaries_by_id[did].key_findings) + len(summaries_by_id[did].quotes)),
            ),
        )
        top_doc_ids = ranked_doc_ids[:MAX_SUMMARIES_P_SEC]
        section_summaries = [summaries_by_id[did] for did in top_doc_ids]
 
        sends.append(Send("write_section", {
            "task_id": state["task_id"],
            "section": section,
            "claims": section_claims,
            "doc_summaries": section_summaries,
            "original_query": state["original_query"],
        }))
    return sends



WRITE_SECTION_PROMPT = """You are writing ONE section of a research report.


Research question: {question}
Section title: {title}
Section angle: {angle}

You have two inputs:
1. Claims: Small factual statements for this section. This tells you WHAT this section should cover.
2. Supporting source summaries: evidence and context from the documents behind those claims. Each has:
    - doc_id: the citation key (cite as [d_<doc_id>], e.g. [d_fb2843aa4730])
    - overview: what the document is and how it frames the topic 
    - key_findings: specific facts (dates, numbers, named people/places/events)
    - quotes: short verbatim phrases you may incorporate

Claims:
{claims_json}

Supporting source summaries:
{doc_summaries_json}

Rules:
- Write 200- 800 words of flowing prose for a curious reader.
- DO NOT attempt to include section title or any heading in the output — just write the body. 
- Connect facts with reasoning words: 'because', 'therefore', 'however', 
'consequently', etc. Don't just list facts in a sequence.
- Use the claims as the main factual structure; use the supporting summaries for context, chronology, and causal relationships.
- Cite each fact inline using the doc_id of the source it came from, in brackets with a 'd_' prefix: [d_fb2843aa4730]. Multiple sources for one fact: [d_fb2843aa4730, d_512264222188]. Place the citation immediately after the sentence or clause it supports.
- The overview tells you what each source actually argues — use that to attribute interpretations to specific sources rather than asserting them as fact.
-Be specific about names, dates and figures. Avoid generalizations and hedging.
-Use only facts present in the provided claims. Do not add names, dates, or details not in the claims, even if you know them."
-If the evidence does not support part of the section angle, omit that point instead of inventing analysis.
-Do not end the section with a summary paragraph; stop when the content is covered.

Output markdown only."""


async def write_section(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    section: SectionPlan = branch_input["section"]
    claims: list[Claim] = branch_input["claims"]
    doc_summaries: list[DocSummary] = branch_input["doc_summaries"]
    
    _emit(task_id, "writing_section", title=section.title)
    
    claims_json = json.dumps([c.model_dump() for c in claims])
    summaries_json = json.dumps(
        [
            {
                "doc_id": f"d_{s.doc_id}",  # writer cites this verbatim as [d_xxx]
                "title": s.title,
                "overview": s.overview,
                "key_findings": s.key_findings,
                "quotes": s.quotes,
            }
            for s in doc_summaries
        ],
        ensure_ascii=False,
    )
    llm = get_clients().writer_llm(temperature=0.5, max_tokens=4096)
    
    resp = await llm.ainvoke([
        {"role": "user", "content": WRITE_SECTION_PROMPT.format(
            question=branch_input["original_query"],
            title=section.title,
            angle=section.angle,
            claims_json=claims_json,
            doc_summaries_json=summaries_json
        )},
    ])
    body = resp.content.strip()
    
    # Remove thinking blocks
    body = re.sub(r"<think>.*?<think>", "", body, flags=re.DOTALL).strip()
    if "<tool_call>" in body:
        logger.error("Writer leaked incomplete reasoning for section: %s", section.title)
        body = body.split("<think>", 1)[0].strip()

    # Match single OR grouped citations: [d_xxx] and [d_xxx, d_yyy]. Stays in lockstep
    # with search._CITE_RE so the audit field matches what the stitcher resolves.
    cited_groups = re.findall(r"\[(d_[a-f0-9]+(?:\s*,\s*d_[a-f0-9]+)*)\]", body)
    citations_used = list(dict.fromkeys(
        cid.strip() for group in cited_groups for cid in group.split(",")
    ))

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
    
    # Walk the actual citation markers, not citations_used metadata — markers are truth
    ordered_doc_ids = _collect_ordered_doc_ids(ordered)
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
        rewritten = _rewrite_citations(sec.body_markdown, doc_to_ref)
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