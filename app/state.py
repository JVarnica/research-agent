
from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel, Field


class QueryLLM(BaseModel):
    """A search query with a one-line rationale. The rationale forces
    the model to commit to *why* it's running this query."""
    query: str = Field(
        min_length=4, max_length=200,
        description=(
            "A focused noun phrase or specific question, 5-15 words. "
            "Include a concrete qualifier like 'paper', 'timeline', 'causes', "
            "'documentation', or 'explained'. "
            "Avoid bare keywords, vague phrasing, and weak words "
            "(thing, stuff, good, bad, use). "
            "Good: 'multi-head attention transformer explained'. "
            "Good: 'Roman Empire collapse causes'. "
            "Bad: 'multi head attention function' (bare keywords, no qualifier). "
            "Bad: 'what was the Roman Empire' (vague, no angle)."
        ),
    )
    rationale: str = Field(
        description="One sentence: why this query advances the research.",
        max_length=200,
    )
    category: Literal["general", "science", "it", "news"] = Field(
        default="general",
        description=(
            "SearXNG routing. "
            "'science': academic, medicine, biology, chemistry, physics (arxiv, pubmed). "
            "'it': programming, software, technical documentation. "
            "'general': history, business, culture, everything non-technical. "
            "'news': current events only."
        ),
    )

class Query(QueryLLM):
    id: str 

class QuerySetLLM(BaseModel):
    """A diverse set of queries covering different angles of the question."""
    queries: list[QueryLLM] = Field(
        min_length=2, max_length=5,
        description=(
            "3-5 queries, each exploring a distinct angle. "
            "No near-duplicates — two queries that differ only in wording count as one."
        ),
    )

class Document(BaseModel):
    """A raw search hit with scraped content. Cold storage — never sent 
    whole to the LLM. The summarizer reads it one at a time."""
    id: str
    url: str
    title: str
    content: str
    source_query_id: str
    search_score: float = 0.0


class DocSummaryLLM(BaseModel):
    """Compressed view of a single doc.
    Schema fields are deliberate: `relevant` forces a yes/no commitment, 
    `key_findings` forces specificity, `quotes` provide auditable evidence."""
    relevant: bool = Field(description="Does this doc actually help answer the question?")
    overview: str = Field(
        default="",
        description=(
            " 1-3 sentences: what this document covers, its angle, scope, or argument. "
            "Helps the section writer use this source in context rather than as "
            "disconnected facts. Empty string if not relevant."
        ),
        max_length = 400
    )
    key_findings: list[str] = Field(
        description="3-8 specific findings, each stating a date, number, named person/place/event. No more than 3 sentences the finding. Empty list if not relevant.",
        max_length=8,
    )
    quotes: list[str] = Field(
        description="Up to 4 short supporting quotes backing a finding. Optional (<25 words each)",
        max_length=4,
    )

class DocSummary(DocSummaryLLM):
    doc_id: str
    title: str
    url: str

def merge_claims_by_topic(
    existing: list[Claim],
    updates: list[Claim],
) -> list[Claim]:
    by_topic = {claim.topic: claim for claim in existing}

    for update in updates:
        previous = by_topic.get(update.topic)

        if previous is None:
            by_topic[update.topic] = update
            continue

        by_topic[update.topic] = Claim(
            topic=previous.topic,
            source_doc_ids=list(
                dict.fromkeys([
                    *previous.source_doc_ids,
                    *update.source_doc_ids,
                ])
            ),
            confidence=update.confidence,
        )

    return list(by_topic.values())

class Claim(BaseModel):
    """A specific factual claim with provenance. Built by aggregating 
    findings across docs. The planner organizes these into sections."""
    topic: str = Field(
        description=("Canonical topic label. Reuse the exact existing topic label "
            "when new evidence belongs to an existing topic."),
        max_length=150)
    source_doc_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    
class SectionPlanLLM(BaseModel):
    """A planned section. `claim_ids` is mp and summaries is the detail."""
    title: str
    angle: str = Field(
        description="2 sentences about what this section covers. Not the content itself.", 
        max_length=300)
    claim_topics: list[str] = Field(
        min_length=1,
        description=(
            "Extact topic labels copied from the provided claims."
            "Do not paraphrase or invent topic labels."
        ),
    )
class SectionPlan(SectionPlanLLM):
    id: str

class ReportPlanLLM(BaseModel):
    title: str = Field(max_length=150)
    sections: list[SectionPlanLLM] = Field(min_length=3, max_length=6)

class ReportPlan(BaseModel):
    title: str = Field(max_length=150)
    sections: list[SectionPlan] = Field(min_length=3, max_length=6)

class WrittenSection(BaseModel):
    """A section after the writer has filled it in. `citations_used` lets 
    the stitcher build the reference list and lets us audit grounding."""
    id: str
    title: str
    body_markdown: str
    citations_used: list[str] = Field(default_factory=list)


class ReflectionResultLLM(BaseModel):
    current_understanding: str = Field(
        description=(
            "2-5 sentences summarizing what the claims have established so far."
            "Be specific about concrete findings, not vague"
        ),
        max_length=800 #kept truncating at 400
    )
    is_sufficient: bool
    knowledge_gap: str = Field(description="What's still missing, 2-3 sentences", max_length=400)
    follow_up_queries: list[QueryLLM] = Field(default_factory=list, max_length=4)

class ReflectionResult(ReflectionResultLLM):
    loop: int = 0
    follow_up_queries: list[Query] = Field(default_factory=list)

#Overall state
class OverallState(TypedDict):
    # ---- Input ----
    task_id: str
    original_query: str
    max_research_loops: int

    # ---- Accumulated across the run (parallel-safe via reducers) ----
    search_queries: Annotated[list[Query], operator.add]
    searched_queries_ids: Annotated[list[str], operator.add]
    search_hits: list[Document] # un-scraped content
    hits_to_scrape: list[Document] # overwrite per loop 
    raw_docs: Annotated[list[Document], operator.add] # scraped content
    doc_summaries: Annotated[list[DocSummary], operator.add]
    claims: Annotated[list[Claim], merge_claims_by_topic]
    understanding_history: Annotated[list[str], operator.add] 
    reflection_history: Annotated[list[ReflectionResult], operator.add]
    reflected_doc_ids: Annotated[list[str], operator.add]
    seen_urls: Annotated[list[str], operator.add]
    written_sections: Annotated[list[WrittenSection], operator.add]

    # ---- Loop control (single-writer, overwrite semantics) ----
    research_loop_count: int
    is_sufficient: bool

    # ---- Set mid-run by specific nodes ----
    plan: NotRequired[ReportPlan] #not required optional as first nodes wont ouput anything to it
    final_report: NotRequired[str]
