import asyncio
import json
import logging
import uuid
import re
from typing import Any, Literal
import httpx
from pydantic import BaseModel, Field

from langgraph.types import Send
from .state import (
    OverallState, Query, QuerySetLLM, Document, DocSummaryLLM, DocSummary,
    Claim, ReportPlanLLM, ReportPlan, SectionPlan, WrittenSection, ReflectionResult,
    ReflectionResultLLM 
)
from .llm import get_clients
from .search import (
    search_query, collect_ordered_doc_ids, rewrite_citations, scrape_hits,
    merge_adjacent_numeric_cites
)
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

async def generate_queries(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="generating_queries", message="Planning search strategy")
    
    structured = get_clients().struct_cheap_llm(QuerySetLLM)
    result: QuerySetLLM = await structured.ainvoke([
        {"role": "system", "content": QUERY_GEN_PROMPT},
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

# ============================================================
# Node: fan-out summarization — one branch per doc
# ============================================================

def fan_out_summarize(state: OverallState) -> list[Send]:
    summarized_ids = {s.doc_id for s in state.get("doc_summaries", [])}
    pending = [d for d in state["raw_docs"] if d.id not in summarized_ids]
    if not pending:
        return "extract_claims" # skip summary no docs
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
- If relevant=false, return an empty overview, key_findings, and quotes.

Rules:
- overview: 1-2 sentences orienting a downstream writer on what THIS document is — its angle, scope, or argument. Examples across domains: "Wikipedia survey of the topic, organised chronologically", "Peer-reviewed study (n=2,400) arguing X is the primary cause, not Y", "Vendor whitepaper promoting their product; useful for benchmark numbers but biased", "First-person blog post from a practitioner; anecdotal not statistical". This is NOT a recap of the facts — it's the document's shape and stance. Skip if not relevant.
- key_findings: 3-8 findings. DO NOT invent findings to reach count. 2 sharp findings are better than 3 padded ones.
- Each finding must state a specific date, number, named person/place/event or causal mechanism. 
- A causal-mechanism finding states cause then effect (X caused Y because Z), not a vague description.
- Do not hedge or generalize. Quote specifics.
- quotes: up to 4 short verbatim quotes (under 25 words each) that support your findings. Omit if add nothing.

Document title: {title}
URL: {url}

Document content:
{content}"""

async def summarize_one_doc(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    doc: Document = branch_input["doc"]
    
    try:
        # validation & retry guided decoding affects too much
        result: DocSummaryLLM = await get_clients().validated(DocSummaryLLM,
            [{"role": "user", "content": SUMMARIZE_PROMPT.format(
                question=branch_input["original_query"],
                title=doc.title,
                url=doc.url,
                content=doc.content,
            )},],
            tier="cheap",
            )
        # Add the document IDS 
        summary = DocSummary(
            doc_id = doc.id,
            url = doc.url,
            title = doc.title,
            **result.model_dump()
        )
        logger.debug("summary doc_id=%s url=%s relevant=%s findings=%s quotes=%s",
                doc.id, doc.url, summary.relevant,
                summary.key_findings, summary.quotes)
    except Exception as e:
        logger.exception(f"summarize failed for {doc.url}: {e}")
        # Return a "not relevant" placeholder so the fan-out completes cleanly
        summary = DocSummary(doc_id=doc.id, url=doc.url, title=doc.title, relevant=False,
                             overview="", key_findings=[], quotes=[])
    
    _emit(task_id, "doc_summarized", doc_title=summary.title, overview=summary.overview, relevant=summary.relevant)
    return {"doc_summaries": [summary]}

# ============================================================
# Node 4: extract claims from all relevant summaries
# ============================================================

EXTRACT_PROMPT = """You are AGGREGATING document findings into topic-level claims.

Research question: {question}

A claim is a TOPIC LABEL plus the doc_ids whose summaries cover that topic. The label
identifies the topic — it is NOT a full account. The actual evidence (dates, quotes,
specifics) stays inside the doc_summaries; the section writer reads those directly.

You receive:
1. EXISTING_CLAIMS — topic claims from earlier loops. Each has topic, source_count, confidence.
2. NEW_SUMMARIES — document summaries you have not yet processed (key_findings + quotes).

For each topic in NEW_SUMMARIES, do exactly ONE of:

(A) MERGE — the topic already appears in EXISTING_CLAIMS (same event, year, and
    entity, even if worded differently).
    - topic: copy the existing topic label VERBATIM, character for character.
    - source_doc_ids: ONLY the new doc_id(s) supporting it. Do not repeat doc_ids
      already counted in the existing claim.
    - Return at most ONE record per existing topic. If multiple new summaries
      support it, combine their doc_ids into one source_doc_ids list.
    - confidence: "high" if existing source_count + new sources >= 3, else "medium".

(B) NEW — no existing claim covers this topic.
    - topic: short label, under ~80 chars where possible. Include the key entity
      and date if applicable. Do NOT write a full explanatory sentence.
    - source_doc_ids: doc_ids from NEW_SUMMARIES supporting this topic.
    - confidence: high (>= 3 sources), medium (1-2), low (sources conflict).

(C) DROP — the topic is too vague to be useful ("various challenges", "industry
    concerns", "long-term trends"). Omit it from the output entirely.

Goal: ~one claim per distinct topic. Prefer fewer well-aggregated claims over many
fragments. Single-source topics are OK if specific.

GOOD CLAIMS (topic labels — mix of domains shown):
- "GPT-4 release, March 2023" (tech / product)
- "CRISPR-Cas9 mechanism: guide RNA + Cas9 cleavage" (science / mechanism)
- "Lehman Brothers bankruptcy, 15 Sep 2008" (finance / event)
- "Treaty of Westphalia 1648: sovereignty principle" (history / concept)
- "Transformer architecture self-attention scaling, O(n²)" (technical / property)

DROP (no entity, no date, no quantity, no named mechanism):
- "There were significant developments in the field."
- "Several factors contributed to the outcome."
- "The technology had wide-ranging impacts."

EXISTING_CLAIMS:
{existing_claims_json}

NEW_SUMMARIES:
{summaries_json}"""


class ClaimSet(BaseModel):
    claims: list[Claim] = Field(max_length=15) #bounding is expensive changed to 15. 


async def extract_claims(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="extracting_claims", message="Aggregating sources into topic claims")

    existing_claims = state.get("claims", [])
    covered_doc_ids = {did for c in existing_claims for did in c.source_doc_ids}
    new_relevant = [
        s for s in state["doc_summaries"]
        if s.relevant and s.doc_id not in covered_doc_ids
    ]
    loop = state.get("research_loop_count", 0)
    if not new_relevant:

        _emit(task_id, "claims_extracted", count=0, loop=loop)
        return {"claims": []}

    existing_claims_json = json.dumps(
            [{
                "topic": claim.topic,
                "source_count": len(claim.source_doc_ids),
                "confidence": claim.confidence,
            } 
            for claim in existing_claims
        ], 
        ensure_ascii=False,
    )
    summaries_json = json.dumps(
        [{#don't need url or quotes this is just topic label. overview could be fine in itself.
            "doc_id": s.doc_id,
            "title": s.title,
            "overview": s.overview,
            "key_findings": s.key_findings,
            }
            for s in new_relevant
        ], 
        ensure_ascii=False
    )

    #structured = get_clients().structured_llm(ClaimSet)

    result: ClaimSet = await get_clients().validated(
        ClaimSet,
        [{   
            "role": "user", 
            "content": EXTRACT_PROMPT.format(
                question=state["original_query"],
                existing_claims_json=existing_claims_json,
                summaries_json=summaries_json,
            ),
        },],
        tier="reasoner",
        )
    
    existing_by_topic = {
        claim.topic: claim
        for claim in existing_claims
    }

    valid_new_doc_ids = {
        summary.doc_id
        for summary in new_relevant
    }

    output_claims: list[Claim] = []
    seen_output_topics: set[str] = set()

    new_count = 0
    merged_count = 0

    for generated in result.claims:
        topic = generated.topic.strip()

        if not topic:
            logger.warning("extract_claims: dropped claim with empty topic")
            continue

        if topic in seen_output_topics:
            logger.warning(
                "extract_claims: duplicate topic returned in one call: %s",
                topic,
            )
            continue

        seen_output_topics.add(topic)

        # Only accept document IDs from the new summaries supplied to this call.
        new_source_doc_ids = list(
            dict.fromkeys(
                doc_id
                for doc_id in generated.source_doc_ids
                if doc_id in valid_new_doc_ids
            )
        )

        if not new_source_doc_ids:
            logger.warning(
                "extract_claims: dropped topic '%s' because it had no valid "
                "new source document IDs",
                topic,
            )
            continue

        previous = existing_by_topic.get(topic)

        if previous is not None:
            combined_doc_ids = list(
                dict.fromkeys([
                    *previous.source_doc_ids,
                    *new_source_doc_ids,
                ])
            )

            output_claims.append(
                Claim(
                    topic=previous.topic,
                    source_doc_ids=combined_doc_ids,
                    confidence=generated.confidence,
                )
            )

            merged_count += 1

        else:
            output_claims.append(
                Claim(
                    topic=topic,
                    source_doc_ids=new_source_doc_ids,
                    confidence=generated.confidence,
                )
            )

            new_count += 1

    logger.debug(
        "claims: %d new, %d merged",
        new_count,
        merged_count,
    )
    _emit(task_id, "claims_extracted", count=new_count, merged=merged_count, topics=[claim.topic for claim in output_claims])

    return {"claims": output_claims}


# ============================================================
# Node 5: reflect — do we need another search loop?
# ============================================================

REFLECT_PROMPT = """You are auditing research progress. Decide whether the gathered evidence \
sufficiently answers the research question, or if another search loop is needed.

Research question: 
{question}

Previous reflection from earlier loops:
{reflection_history}

Topics covered so far (claim labels — quick index of what's been touched):
{claims_index}

Evidence already assessed before the latest search loop):
{previous_evidence_json}

New evidence added since the latest search loop:
{new_evidence_json}

Queries already searched (do not repeat or paraphrase these):
{previous_queries}

Budget Line, loop count: 
{budget_line}

Decide whether the evidence is sufficient to write a credible report.


How to judge:
- The claims index shows what topics have been touched. The evidence shows what's
  actually been learned. Judge from the evidence, not the claims.
- Sufficient means the main question can be answered with concrete support drawn
  from the evidence. Not exhaustively — credibly.
- A topic being broad is not a gap. A specific question the report cannot honestly
  answer from the evidence IS a gap.
- New evidence vs previous evidence: ask whether the new evidence resolves a gap
  the previous loop identified, or whether it just adds variety on already-covered
  ground.

If the previous loop generated follow-up queries that returned no new evidence, that gap is unfillable from web search. Don't ask it again — proceed with sufficient=true and acknowledge the limitation in current_understanding.

If you do request follow-ups:
- Search-engine queries, 4-12 words, keyword-dense. Not natural-language questions.
- One concrete gap, 1-3 queries attacking it from genuinely different angles.
- Must not paraphrase queries from previous loops.

Output:
- current_understanding: 2-5 sentences on what the evidence supports. If earlier gaps remain unfilled despite searching, say so.
- is_sufficient: true if the question can be answered credibly OR if remaining gaps are unfillable.
- knowledge_gap: empty if sufficient. Otherwise one concrete unresolved gap.
- follow_up_queries: empty if sufficient. Otherwise 1-3 queries."""

async def reflect(state: OverallState) -> dict:
    task_id = state["task_id"]
    loop = state.get("research_loop_count", 0)
    max_loops = state.get("max_research_loops", 3)
    remaining = max_loops - loop

    budget_line = (
    f"This is research loop {loop + 1} of {max_loops}. "
    + (
        "This is your FINAL loop — no further searches will run. "
        "Set is_sufficient=true and return no follow-up queries."
        if remaining <= 1
        else f"You have {remaining - 1} more loop(s) available after this one."
    )
)

    _emit(task_id, "stage", stage="reflecting", loop=loop)

    relevant_summaries = [
        s for s in state.get("doc_summaries", [])
        if s.relevant
    ]
    alrd_reflected_ids = set(state.get("reflected_doc_ids", []))

    previous_summaries = [
        s for s in relevant_summaries
        if s.doc_id in alrd_reflected_ids
    ]
    new_summaries = [
        s for s in relevant_summaries
        if s.doc_id not in alrd_reflected_ids
    ]
    

    if loop > 0 and not new_summaries:
        logger.warning("reflect: previous loop produced no new evidence — forcing sufficient")
        # Construct a terminal ReflectionResult without an LLM call
        terminal = ReflectionResult(
            loop=loop,
            current_understanding=(
                "Previous follow-up searches returned no new evidence. "
                "Proceeding to report with current claims; the unfilled gap is "
                "acknowledged as a limitation."
            ),
            knowledge_gap="",
            is_sufficient=True,
            follow_up_queries=[],
        )
        _emit(task_id, "reflection", sufficient=True, current_understanding=terminal.current_understanding, knowledge_gap=terminal.knowledge_gap)
        
        return {
        "is_sufficient": True,
        "search_queries": [],  # extends via operator.add
        "research_loop_count": loop + 1,
        "reflection_history": [terminal],  # extends via operator.add
        "understanding_history": [terminal.current_understanding],
        "reflected_doc_ids": [],  # extends via operator.add
    }
        
    claims_index = json.dumps(
        [{"topic": c.topic, "sources": len(c.source_doc_ids),
          "confidence": c.confidence} for c in state["claims"]],
        ensure_ascii=False,
    )
    ### summaries json
    previous_evidence_json = json.dumps(
        [{"doc_id": s.doc_id, 
          "title": s.title,
          "overview": s.overview, 
          "key_findings": s.key_findings}
         for s in previous_summaries],
        ensure_ascii=False,
    )
    new_evidence_json = json.dumps(
        [{"doc_id": s.doc_id, 
          "title": s.title,
          "overview": s.overview, 
          "key_findings": s.key_findings}
         for s in new_summaries],
        ensure_ascii=False,
    )
    # reflection_history
    reflection_history = state.get("reflection_history", [])
    if reflection_history:
        previous_reflections = "\n\n".join(
            f"Loop {i+1} understanding: {r.current_understanding}\n"
            f"Loop {i+1} gap identified: {r.knowledge_gap or '(none — was sufficient)'}\n"
            f"Loop {i+1} follow-ups: {', '.join(q.query for q in r.follow_up_queries) or '(none)'}"
            for i, r in enumerate(reflection_history)
        )
    else:
        previous_reflections = f"(first reflection — no prior history)"
        

    prior_queries = state.get("search_queries", [])
    previous_queries = "\n".join(f"- {q.query}" for q in prior_queries) or "(none yet)"

    #structured = get_clients().structured_llm(ReflectionResultLLM)

    draft: ReflectionResultLLM = await get_clients().validated(
        ReflectionResultLLM,
        [{"role": "user", "content": REFLECT_PROMPT.format(
            question=state["original_query"],
            reflection_history=previous_reflections,
            claims_index=claims_index,
            previous_evidence_json=previous_evidence_json,
            new_evidence_json=new_evidence_json,
            previous_queries=previous_queries,
            budget_line=budget_line
        ),}],
        tier="reasoner"
        )
    
    # Stamp IDs on follow-ups
    new_queries = [
        Query(id=f"q_{uuid.uuid4().hex[:8]}", query=q.query, rationale=q.rationale, category=q.category)
        for q in draft.follow_up_queries
    ]
    
    if not draft.is_sufficient and not new_queries:
        logger.warning("reflect: all follow-ups were duplicates — forcing termination")
    
    result = ReflectionResult(
        current_understanding=draft.current_understanding,
        is_sufficient=draft.is_sufficient or not new_queries,
        knowledge_gap=draft.knowledge_gap,
        follow_up_queries=new_queries,
        loop=loop,
    )

    _emit(task_id, "reflection", sufficient=result.is_sufficient, current_understanding=result.current_understanding, knowledge_gap=result.knowledge_gap,
          new_queries=len(new_queries))
    
    return {
        "is_sufficient": result.is_sufficient,
        "search_queries": new_queries,  # extends via operator.add
        "research_loop_count": loop + 1,
        "reflection_history": [result],  # extends via operator.add
        "understanding_history": [result.current_understanding],
        "reflected_doc_ids": [s.doc_id for s in new_summaries],  # extends via operator.add
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

PLAN_PROMPT = """You are planning a grounded research report.

Research question:
{question}

Final research understanding:
{final_understanding}

Topic claims:
{claims_json}

Evidence catalogue:
{evidence_json}

How to use the inputs:

- Claims are canonical topic handles.
- Each claim topic maps to supporting document summaries.
- The evidence catalogue contains the actual detail behind those topics.
- Use the evidence to decide which topics belong together in a section.
- Do not create a section whose angle is unsupported by the supplied evidence.
- Do not force every minor topic into the report.
- Avoid overlapping sections.

Produce a report plan with 3-5 sections.

Each section must:

- Have a distinct and specific angle.
- Include the exact claim topic labels needed to write that section.
- Copy claim topics exactly, character for character.
- Never paraphrase, shorten, or invent a claim topic.
- Contain only topics supported by the evidence catalogue.

Order sections logically, such as chronologically, causally, or from mechanism
to consequences.

This is a structural plan, not the report.

Output fields:

- title: report title, maximum 12 words
- sections:
  - title: section title
  - angle: one or two sentences describing the section scope
  - claim_topics: exact topic strings copied from Topic claims
"""

async def plan_report(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="planning", message="Structuring the report")

    history = state.get("reflection_history", [])
    final_understanding = history[-1].current_understanding if history else "No reflections yet"
    
    claims_json = json.dumps(
        [{"topic": c.topic,
            "source_doc_ids": c.source_doc_ids, 
            "confidence": c.confidence,
            }
            for c in state["claims"]
        ],
        ensure_ascii=False,
    )
    evidence_json = json.dumps(
        [{"doc_id": s.doc_id, "title": s.title, "overview": s.overview, "key_findings": s.key_findings}
         for s in state["doc_summaries"] if s.relevant],
        ensure_ascii=False,
    )
    structured = get_clients().struct_cheap_llm(ReportPlanLLM)
    
    result: ReportPlan = await structured.ainvoke([
        {"role": "user", 
         "content": PLAN_PROMPT.format(
            question=state["original_query"], final_understanding=final_understanding,
            claims_json=claims_json, evidence_json=evidence_json, 
            ),
        },
    ])
    available_topics = {
        claim.topic
        for claim in state.get("claims", [])
    }

    sections: list[SectionPlan] = []

    for index, generated_section in enumerate(result.sections):
        valid_topics = list(
            dict.fromkeys(
                topic
                for topic in generated_section.claim_topics
                if topic in available_topics
            )
        )

        if not valid_topics:
            logger.warning(
                "plan_report: dropping section '%s' because it contains "
                "no valid claim topics",
                generated_section.title,
            )
            continue

        sections.append(
            SectionPlan(
                id=f"s_{index}",
                title=generated_section.title,
                angle=generated_section.angle,
                claim_topics=valid_topics,
            )
        )

    if not sections:
        raise ValueError(
            "Report planner returned no sections containing valid claim topics"
        )

    plan = ReportPlan(
        title=result.title,
        sections=sections,
    )

    _emit(task_id, "plan_ready", 
          title=plan.title, 
          sections=[{"title": s.title, "claim_count": len(s.claim_topics)} for s in plan.sections],
          )
    return {"plan": plan}


# ============================================================
# Node 7: fan-out section writing
# ============================================================

# Cap on doc_summaries passed to a single section writer call. Most-central
# summaries (those supporting the most claims in the section) are kept first.
# 8 covers richest topics without ballooning prompt size past ~5k evidence tokens.
MAX_SUMMARIES_PER_SECTION = 8


def fan_out_sections(state: OverallState) -> list[Send]:
    claims_by_topic = {c.topic: c for c in state["claims"]}
    summaries_by_id = {
        s.doc_id: s for s in state["doc_summaries"] if s.relevant
    }

    sends: list[Send] = []
    for section in state["plan"].sections:
        section_claims = [
            claims_by_topic[topic] for topic in section.claim_topics if topic in claims_by_topic
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
        top_doc_ids = ranked_doc_ids[:MAX_SUMMARIES_PER_SECTION]
        section_summaries = [summaries_by_id[did] for did in top_doc_ids]

        sends.append(
            Send("write_section", 
                {
                    "task_id": state["task_id"],
                    "section": section,
                    "claims": section_claims,
                    "doc_summaries": section_summaries,
                    "original_query": state["original_query"],
                },
            )
        )
    return sends


WRITE_SECTION_PROMPT = """You are writing ONE section of a grounded research report.

Research question: {question}
Section title: {title}
Section angle: {angle}

You have two inputs:
1. Claims: the topic handles selected for this section. They identify what the section should cover and which supporting summaries were retrieved.
2. Supporting source summaries: evidence and context from the documents behind those claims. Each has:
   - doc_id: the citation key (cite as [d_<doc_id>], e.g. [d_fb2843aa4730])
   - overview: what the document is and how it frames the topic
   - key_findings: specific facts (dates, numbers, named people/places/events)
   - quotes: short verbatim phrases you may incorporate

Claims:
{claims_json}

Supporting source summaries:
{summaries_json}

Rules:
- Write only the body of the section. Do not include a heading.
- Use the claims as the main factual structure; use the supporting summaries for context, chronology, and causal relationships.
- Cite each fact inline using the doc_id of the source it came from, in brackets with a 'd_' prefix: [d_fb2843aa4730]. Multiple sources for one fact: [d_fb2843aa4730, d_512264222188]. Place the citation immediately after the sentence or clause it supports.
- The overview tells you what each source actually argues — use that to attribute interpretations to specific sources rather than asserting them as fact.
- Do not introduce a factual detail unless it appears in either the claims or the supporting source summaries.
- Do not infer a causal relationship merely because two events appear together.
- If the evidence does not support part of the section angle, omit that point instead of inventing analysis.
- Avoid repeating facts already covered by the section unless needed for the argument.
- Write 150-500 words. Do not pad thin evidence into a long section.
- Do not include a concluding summary paragraph.

Output markdown only."""


async def write_section(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    section: SectionPlan = branch_input["section"]
    claims: list[Claim] = branch_input["claims"]
    doc_summaries: list[DocSummary] = branch_input["doc_summaries"]

    _emit(task_id, "writing_section", title=section.title)

    claims_json = json.dumps(
        [c.model_dump() for c in claims],
        ensure_ascii=False,
    )
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

    llm = get_clients().writer_llm()
    resp = await llm.ainvoke([
        {
            "role": "user",
            "content": WRITE_SECTION_PROMPT.format(
                question=branch_input["original_query"],
                title=section.title,
                angle=section.angle,
                claims_json=claims_json,
                summaries_json=summaries_json,
            ),
        },
    ])
    body = resp.content.strip()

    if not body or len(body) < 50:
        logger.error(f"Section writing failed or was too short for section '{section.title}'")

    # Match single OR grouped citations: [d_xxx] and [d_xxx, d_yyy]. Stays in lockstep
    # with search._CITE_RE so the audit field matches what the stitcher resolves.
    cited_groups = re.findall(
        r"\[(d_[a-f0-9]{12}(?:\s*,\s*d_[a-f0-9]{12})*)\]",
        body,
    )
    citations_used = list(dict.fromkeys(
        cid.strip() for group in cited_groups for cid in group.split(",")
    ))

    written = WrittenSection(
        id=section.id,
        title=section.title,
        body_markdown=body,
        citations_used=citations_used,
    )
    _emit(
        task_id,
        "section_written",
        title=section.title,
        citations=len(citations_used),
    )
    return {"written_sections": [written]}


# ============================================================
# Node 8: stitch sections into the final report
# ============================================================

async def stitch_report(state: OverallState) -> dict:
    task_id = state["task_id"]

    _emit(
        task_id,
        "stage",
        stage="stitching",
        message="Assembling final report",
    )

    sections_by_id = {
        section.id: section
        for section in state["written_sections"]
    }

    ordered_sections = [
        sections_by_id[planned.id]
        for planned in state["plan"].sections
        if planned.id in sections_by_id
    ]

    missing_section_ids = [
        planned.id
        for planned in state["plan"].sections
        if planned.id not in sections_by_id
    ]

    if missing_section_ids:
        logger.warning(
            "Missing written sections during stitching: %s",
            missing_section_ids,
        )

    docs_by_id = {
        document.id: document
        for document in state["raw_docs"]
    }

    cited_doc_ids = collect_ordered_doc_ids(ordered_sections)

    invalid_doc_ids = [
        doc_id
        for doc_id in cited_doc_ids
        if doc_id not in docs_by_id
    ]

    if invalid_doc_ids:
        logger.warning(
            "Unknown document IDs cited by section writers: %s",
            invalid_doc_ids,
        )

    ordered_doc_ids = [
        doc_id
        for doc_id in cited_doc_ids
        if doc_id in docs_by_id
    ]

    doc_to_ref = {
        doc_id: index
        for index, doc_id in enumerate(ordered_doc_ids, start=1)
    }

    references = [
        (
            f"{doc_to_ref[doc_id]}. "
            f"[{docs_by_id[doc_id].title or docs_by_id[doc_id].url}]"
            f"({docs_by_id[doc_id].url})"
        )
        for doc_id in ordered_doc_ids
    ]

    intro_conclusion = await _write_intro_conclusion(
        state,
        ordered_sections,
    )

    report_parts = [
        f"# {state['plan'].title}",
        "",
        intro_conclusion["intro"],
        "",
    ]

    for section in ordered_sections:
        rewritten = rewrite_citations(
            section.body_markdown,
            doc_to_ref,
        )
        rewritten = merge_adjacent_numeric_cites(rewritten)

        report_parts.extend([
            f"## {section.title}",
            "",
            rewritten,
            "",
        ])

    report_parts.extend([
        "## Conclusion",
        "",
        intro_conclusion["conclusion"],
        "",
        "## References",
        "",
        *references,
    ])

    final_report = "\n".join(report_parts)

    _emit(
        task_id,
        "report_ready",
        length=len(final_report),
    )

    return {"final_report": final_report}

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
    structured = get_clients().structured_llm(IntroConclusion)
    result: IntroConclusion = await structured.ainvoke([
        {"role": "user", "content": (
            f"Write an intro and conclusion for a report answering: {state['original_query']}\n\n"
            f"The report has these sections:\n{section_summaries}\n\n"
            f"Intro frames the question (2-3 sentences). Conclusion summarizes the answer (2-3 sentences). "
            f"Do not introduce new facts."
        )},
    ])
    return {"intro": result.intro, "conclusion": result.conclusion}