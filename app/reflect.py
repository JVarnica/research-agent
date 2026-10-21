import logging
import uuid
import json

from .task import _emit
from .llm import get_clients
from .state import OverallState, ReflectionResult, ReflectionResultLLM, Query
from .prompts import REFLECT_PROMPT
from .observability import capture_node



logger = logging.getLogger(__name__)

#@capture_node("reflect")
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
    alrd_reflected_ids = set(state.get("reflected_doc_ids", []))

    claims_by_topic = {c.topic: c for c in state.get("claims", [])}
    claims = list(claims_by_topic.values())
    new_doc_ids = {
        did for c in claims for did in c.source_doc_ids
        if did not in alrd_reflected_ids
    }


    if loop > 0 and not new_doc_ids:
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

    def _status(c) -> str:
        unseen = [d for d in c.source_doc_ids if d not in alrd_reflected_ids]
        if not unseen:
            return "seen"
        return "new" if len(unseen) == len(c.source_doc_ids) else "updated"
    
    claims_index = json.dumps(
        [{"topic": c.topic, 
          "sources": len(c.source_doc_ids),
          "confidence": c.confidence,
          "status": _status(c)}
          for c in claims],
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


    doc_summaries = state.get("doc_summaries", [])
    clients = get_clients()

    def evidence_entry(d):
        return {
                "doc_id": d.doc_id,
                "overview": d.overview,
                "key_findings": d.key_findings,
            }
    messages, included_new_ids, omitted_count = await build_reflection_messages(
        clients,
        ReflectionResultLLM,
        REFLECT_PROMPT,
        fields={
            "question": state["original_query"],
            "reflection_history": previous_reflections,
            "claims_index": claims_index,
            "previous_queries": previous_queries,
            "budget_line": budget_line,
        },
        previous_evidence=[
            evidence_entry(d)
            for d in doc_summaries
            if d.doc_id in alrd_reflected_ids
        ],
        new_evidence=[
            evidence_entry(d)
            for d in doc_summaries
            if d.doc_id in new_doc_ids
        ],
        tier="cheap",
    )

    logger.info(
        "reflect: included %d new documents; omitted %d documents due to budget",
        len(included_new_ids),
        omitted_count,
    )

    draft: ReflectionResultLLM = await clients.validated(
        ReflectionResultLLM,
        messages,
        tier="cheap",
        node="reflect",
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
        "reflected_doc_ids": included_new_ids,  # extends via operator.add
    }


def reflect_router(state: OverallState) -> str:
    """Decide: another search loop, or move to planning?"""
    if state["is_sufficient"]:
        return "plan_report"
    if state["research_loop_count"] >= state["max_research_loops"]:
        return "plan_report"
    return "search_all_queries"

async def build_reflection_messages(clients, schema, prompt, *, fields,
                                    previous_evidence, new_evidence,
                                    tier='reasoner', budget=None):
    # New evidence has first priority. Caller chooses relevance ordering within groups.
    records = ([{'group': 'new', 'evidence': d} for d in new_evidence]
               + [{'group': 'previous', 'evidence': d} for d in previous_evidence])
    total = len(records)

    def build(selected):
        previous = [r['evidence'] for r in selected if r['group'] == 'previous']
        new = [r['evidence'] for r in selected if r['group'] == 'new']
        text = prompt.format(
            **fields,
            previous_evidence_json=json.dumps(previous, ensure_ascii=False),
            new_evidence_json=json.dumps(new, ensure_ascii=False),
        )
        omitted = total - len(selected)
        text += (f'\nEvidence budget: {len(selected)} of {total} document records supplied; '
                 f'{omitted} omitted from this view. Omitted documents remain available '
                 'for report writing. Absence from this view is not proof of missing evidence.')
        return [{'role': 'user', 'content': text}]

    selected, omitted = await clients.pack_records(
        schema, records, build, node='reflect', tier=tier, budget=budget)
    included_new_ids = [r['evidence']['doc_id'] for r in selected if r['group'] == 'new']
    return build(selected), included_new_ids, len(omitted)