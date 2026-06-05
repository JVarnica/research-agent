
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
            # long phrases don'twork too many matched key words
            "Search engine query: 4-12 words, keyword style"
            "NOT natural language questions"
            "Strip filler words like 'what were', 'how did', 'the role of'. "
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
    overview: str = Field(
        default="",
        description=(
            " 1-3 sentences: what this document covers, its angle, scope, or argument. "
            "Helps the section writer use this source in context rather than as "
            "disconnected facts. Empty string if not relevant."
        ),
        max_length=300,

    )
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
        description=(
            "Short topic label identifying what this claim covers. Include the "
            "key entity and date or quantity when applicable. "
            "Good: 'GPT-4 release, March 2023'. "
            "Good: 'CRISPR-Cas9 mechanism: guide RNA + Cas9 cleavage'. "
            "Good: 'Lehman Brothers bankruptcy, 15 Sep 2008'. "
            "Not a full explanation — the rich detail lives in the source doc_summaries."
        ),
        max_length=150)
    source_doc_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]

class SectionPlan(BaseModel):
    """A planned section. The writer receives the claims listed here PLUS the
    doc_summaries those claims point to (ranked by centrality, capped per section).
    The angle and claim_ids together define what the writer should cover."""
    id: str 
    title: str = Field(max_length=150)
    angle: str = Field(
        description=(
            "What this section argues or covers, in 2-4 sentences. "
            "Be specific about the angle — what the section establishes, in what order, "
            "and what conclusion it builds toward. "
            "Good: 'Traces the chronology of the 2008 collapse from the Bear Stearns "
            "rescue in March through the Lehman bankruptcy on 15 September. Establishes "
            "that regulatory inaction at three specific decision points enabled the cascade.' "
            "Bad: 'Covers the 2008 financial crisis.' (generic, no angle)"
        ),
        max_length=400
    )
    claim_ids: list[str] = Field(min_length=1, max_length=15)

class ReportPlan(BaseModel):
    title: str = Field(max_length=150)
    sections: list[SectionPlan] = Field(min_length=3, max_length=6)

class ExtractedClaim(BaseModel):
    """A new claim or a merge into an existing one."""
    statement: str = Field(max_length=150)
    source_doc_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
    merges_into: str | None = Field(
        default=None,
        description="Existing claim id (e.g. 'c_5918d72d') if this restates that claim. Null for new claims.",
    )

def merge_claims_by_id(existing: list["Claim"], updates: list["Claim"]) -> list["Claim"]:
    """Reducer for `claims`: an update with a matching id REPLACES the existing
    entry — used so extract_claims can fold new sources into a claim across loops.
    New ids are appended. Preserves insertion order."""
    by_id: dict[str, "Claim"] = {c.id: c for c in existing}
    for c in updates:
        by_id[c.id] = c
    return list(by_id.values())


class WrittenSection(BaseModel):
    """A section after the writer has filled it in. `citations_used` lets 
    the stitcher build the reference list and lets us audit grounding."""
    id: str
    title: str
    body_markdown: str
    citations_used: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    loop: int
    current_understanding: str = Field(
        description=(
            "Few sentences summarizing what the claims have established so far."
            "Be specific about concrete findings, not vague"
        ),
        max_length=800 # kept trunctating at 400
    )
    knowledge_gap: str = Field(description="What's still missing look, 2-3 sentences", max_length=600)
    is_sufficient: bool
    follow_up_queries: list[Query] = Field(default_factory=list, max_length=4)


#Overall state
class OverallState(TypedDict):
    # ---- Input ----
    task_id: str
    original_query: str
    max_research_loops: int

    # ---- Accumulated across the run (parallel-safe via reducers) ----
    search_queries: Annotated[list[Query], operator.add]
    searched_queries_ids: Annotated[list[str], operator.add]
    search_hits: list[SearchHit]
    hits_to_scrape: list[SearchHit] # overwrite per loop 
    raw_docs: Annotated[list[Document], operator.add]
    doc_summaries: Annotated[list[DocSummary], operator.add]
    understanding_history: Annotated[list[str], operator.add]
    claims: Annotated[list[Claim], merge_claims_by_id]
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
