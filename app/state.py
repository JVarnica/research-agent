
from __future__ import annotations
import operator
from typing import TypedDict, Annotated, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel, Field

class Query(BaseModel):
    """A search query with a one-line rationale. The rationale forces
    the model to commit to *why* it's running this query."""
    id: str
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
class SearchHit(BaseModel):
    """Searxng result before scraping — title + snippet only.
    The pre_scrape node decides which of these are worth fetching."""
    id: str
    url: str
    title: str
    snippet: str = ""          # searxng's 'content' field
    source_query_id: str
    search_score: float = 0.0
    category: str = "general"  # carried through for downstream use

class HitVerdict(BaseModel):
    """Per-hit decision. Forcing a verdict on EVERY hit — not a keep-list —
    stops the model defaulting to inclusion by omission."""
    id: str
    keep: bool = Field(
        description="True ONLY if the title/snippet shows the page directly addresses the research question."
    )
    reason: str = Field(
        max_length=250,
        description="Brief reason: e.g. 'dictionary entry for unrelated word' or 'covers Manzikert 1071 directly'.",
    )

class Document(BaseModel):
    """A raw search hit with scraped content. Cold storage — never sent 
    whole to the LLM. The summarizer reads it one at a time."""
    id: str
    url: str
    title: str
    raw_content: str
    source_query_id: str
    search_score: float = 0.0


class DocSummary(BaseModel):
    """Compressed view of a single doc.
    Schema fields are deliberate: `relevant` forces a yes/no commitment, 
    `key_findings` forces specificity, `quotes` provide auditable evidence."""
    doc_id: str
    title: str
    url: str
    relevant: bool = Field(description="Does this doc actually help answer the question?")
    key_findings: list[str] = Field(
        description="3-8 specific findings, each stating a date, number, named person/place/event. No more than 3 sentences the finding. Empty list if not relevant.",
        max_length=8,
    )
    quotes: list[str] = Field(
        description="Up to 4 short supporting quotes backing a finding. Optional (<25 words each)",
        max_length=4,
    )


class Claim(BaseModel):
    """A specific factual claim with provenance. Built by aggregating 
    findings across docs. The planner organizes these into sections."""
    id: str
    statement: str = Field(
        description=("ONE atomic fact in one sentence. Must contain at least one specific: "
            "date, number, named person, named place, or named event. "
            "If the sentence has two independent facts joined by 'and' or a comma, "
            "split it into two claims. "
            "Good: 'Constantinople fell to Mehmed II on 29 May 1453.' "
            "Bad: 'The empire declined due to military weakness, economic stagnation, "
            "and religious schism' (three claims fused into one)."),
        max_length=150)
    source_doc_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class SectionPlan(BaseModel):
    """A planned section. `claim_ids` is the contract — the section writer 
    only gets these claims, nothing else."""
    id: str
    title: str
    angle: str = Field(description="What this section argues or covers in 2-3 sentences", max_length=300)
    claim_ids: list[str] = Field(min_length=1)


class ReportPlan(BaseModel):
    title: str
    sections: list[SectionPlan] = Field(min_length=3, max_length=6)


class WrittenSection(BaseModel):
    """A section after the writer has filled it in. `citations_used` lets 
    the stitcher build the reference list and lets us audit grounding."""
    id: str
    title: str
    body_markdown: str
    citations_used: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    current_understanding: str = Field(
        description=(
            "2-5 sentences summarizing what the claims have established so far."
            "Be specific about concrete findings, not vague"
        ),
        max_length=400
    )
    is_sufficient: bool
    knowledge_gap: str = Field(description="What's still missing, 2-3 sentences", max_length=400)
    follow_up_queries: list[Query] = Field(default_factory=list, max_length=4)


#Overall state
class OverallState(TypedDict):
    # ---- Input ----
    task_id: str
    original_query: str
    max_research_loops: int

    # ---- Accumulated across the run (parallel-safe via reducers) ----
    search_queries: Annotated[list[Query], operator.add]
    search_hits: list[SearchHit]
    hits_to_scrape: list[SearchHit] # overwrite per loop 
    raw_docs: Annotated[list[Document], operator.add]
    doc_summaries: Annotated[list[DocSummary], operator.add]
    claims: Annotated[list[Claim], operator.add]
    understanding_history: Annotated[list[str], operator.add] 
    seen_urls: Annotated[list[str], operator.add]
    written_sections: Annotated[list[WrittenSection], operator.add]

    # ---- Loop control (single-writer, overwrite semantics) ----
    research_loop_count: int
    is_sufficient: bool

    # ---- Set mid-run by specific nodes ----
    plan: NotRequired[ReportPlan] #not required optional as first nodes wont ouput anything to it
    final_report: NotRequired[str]
