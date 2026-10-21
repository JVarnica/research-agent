
from datetime import date
import json
import logging
import re
from collections import Counter
from pydantic import BaseModel, Field
from langgraph.types import Send
from .state import (
    OverallState, DocSummary,
    Claim, ReportPlanLLM, ReportPlan, SectionPlan, WrittenSection
)
from .llm import get_clients
from .search import (
    collect_ordered_doc_ids, rewrite_citations,merge_adjacent_numeric_cites
)
from .events import get_channel
from .prompts import PLAN_PROMPT, WRITE_SECTION_PROMPT
from .task import _emit, today_str

logger = logging.getLogger(__name__)

async def plan_report(state: OverallState) -> dict:
    task_id = state["task_id"]
    _emit(task_id, "stage", stage="planning", message="Structuring the report")

    clients = get_clients()

    history = state.get("reflection_history", [])
    final_understanding = history[-1].current_understanding if history else "No reflections yet"

    claims_json = json.dumps(
        [{"topic": c.topic, "source_doc_ids": c.source_doc_ids, "confidence": c.confidence}
         for c in state["claims"]],
        ensure_ascii=False,
    )

    support = Counter(did for c in state["claims"] for did in c.source_doc_ids)
    evidence = sorted(
        ({"doc_id": s.doc_id, "title": s.title, "overview": s.overview, "key_findings": s.key_findings}
         for s in state["doc_summaries"] if s.relevant),
        key=lambda e: -support[e["doc_id"]],   # stable: ties keep original order
    )
    total = len(evidence)

    def build(batch):
        text = PLAN_PROMPT.format(
            question=state["original_query"],
            current_date=today_str(),
            final_understanding=final_understanding,
            claims_json=claims_json,
            evidence_json=json.dumps(batch, ensure_ascii=False),
        )
        if len(batch) < total:
            text += (f"\nEvidence catalogue shows {len(batch)} of {total} documents for length. "
                     "The claims list is complete; any claim topic may still be used.")
        return [{"role": "user", "content": text}]

    batch, omitted = await clients.pack_records(
        ReportPlanLLM, evidence, build, node="plan_report", tier="cheap")
    if omitted:
        logger.info("plan_report: %d of %d evidence records omitted for budget", len(omitted), total)

    result: ReportPlanLLM = await clients.validated(
        ReportPlanLLM, build(batch), tier="cheap", node="plan_report")

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

async def write_section(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    section: SectionPlan = branch_input["section"]
    claims: list[Claim] = branch_input["claims"]
    doc_summaries: list[DocSummary] = branch_input["doc_summaries"]
    _emit(task_id, "writing_section", title=section.title)

    clients = get_clients()
    claims_json = json.dumps([c.model_dump() for c in claims], ensure_ascii=False)
    records = [
        {"doc_id": f"d_{s.doc_id}", "title": s.title, "overview": s.overview,
         "key_findings": s.key_findings, "quotes": s.quotes}
        for s in doc_summaries
    ]

    def build(batch):
        return [{"role": "user", "content": WRITE_SECTION_PROMPT.format(
            question=branch_input["original_query"],
            current_date=today_str(),
            title=section.title,
            angle=section.angle,
            claims_json=claims_json,
            summaries_json=json.dumps(batch, ensure_ascii=False),
        )}]

    # No schema here, so use the budgeter directly (clients.pack_records would inject one).
    batch, dropped = await clients.budgeter.pack_records(records, build, node="write_section")
    if dropped:
        logger.info("write_section '%s': dropped %d summaries for budget", section.title, len(dropped))

    resp = await clients.budgeted_invoke(clients.writer_llm(), build(batch), node="write_section")
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

_CITE_STRIP_RE = re.compile(r"\s*\[d_[a-f0-9]{12}(?:\s*,\s*d_[a-f0-9]{12})*\]")

async def _write_intro_conclusion(state: OverallState, sections: list[WrittenSection]) -> dict:
    section_summaries = "\n".join(
        f"- {s.title}: {_CITE_STRIP_RE.sub('', s.body_markdown)[:400].rstrip()}..."
        for s in sections
    )
    messages = [{"role": "user", "content": (
            f"Write an intro and conclusion for a report answering: {state['original_query']}\n\n"
            f"The report has these sections:\n{section_summaries}\n\n"
            f"Intro frames the question (2-3 sentences). Conclusion summarizes the answer (2-3 sentences). "
            f"Do not introduce new facts."
        )}]
    
    result: IntroConclusion = await get_clients().validated(
        IntroConclusion,
        messages,
        tier="fast",
        node="intro_conclusion",
    )
    return {"intro": result.intro, "conclusion": result.conclusion}