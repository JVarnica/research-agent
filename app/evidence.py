import logging
import uuid
import json

from langgraph.types import Send
from pydantic import BaseModel, Field, field_validator

from .task import _emit
from .llm import get_clients
from .state import OverallState, Document, DocSummary, DocSummaryLLM, Claim
from .prompts import SUMMARIZE_PROMPT, EXTRACT_PROMPT
from .observability import capture_node

logger = logging.getLogger(__name__)


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


#@capture_node("summarize_one_doc")
async def summarize_one_doc(branch_input: dict) -> dict:
    task_id = branch_input["task_id"]
    doc: Document = branch_input["doc"]
    stats: dict = {}
    
    try:
        result: DocSummaryLLM = await get_clients().validated(
            DocSummaryLLM,
            [{"role": "user", "content": SUMMARIZE_PROMPT.format(
                question=branch_input["original_query"],
                title=doc.title,
                url=doc.url,
                content=doc.content,
            )}],
            tier="cheap",
            node="summarize_one_doc",
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
    
    _emit(task_id, "doc_summarized", title=summary.title, url=summary.url, overview=summary.overview, relevant=summary.relevant)
    return {"doc_summaries": [summary]}

# ============================================================
# Node 4: extract claims from all relevant summaries
# ============================================================


class ClaimSet(BaseModel):
    claims: list[Claim] = Field(description="List of up to 10 distinct topic claims") 

    @field_validator("claims")
    @classmethod
    def _claim_cap(cls, v: list[Claim]) -> list[Claim]:
        return v[:10]  # Limit to 10 claims


#@capture_node("extract_claims")
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

    existing_by_topic = {c.topic: c for c in existing_claims}
    pending = [
        {"doc_id": s.doc_id, "title": s.title, "overview": s.overview}
        for s in new_relevant
    ]
    output_claims: list[Claim] = []
    new_count = 0
    merged_count = 0
    clients = get_clients()
    while pending:
        existing_claims_json = json.dumps(
                [{
                    "topic": claim.topic,
                    "source_count": len(claim.source_doc_ids),
                    "confidence": claim.confidence,
                } 
                for claim in existing_by_topic.values()
        ], ensure_ascii=False)
        
        def build_messages(batch):
            return [{"role": "user", "content": EXTRACT_PROMPT.format(
                question=state["original_query"],
                existing_claims_json=existing_claims_json,
                summaries_json=json.dumps(batch, ensure_ascii=False),
            )}]

        batch, pending = await clients.pack_records(ClaimSet, pending, build_messages, 
                            node="extract_claims", tier="fast")
        result = await clients.validated(ClaimSet, build_messages(batch), tier="fast", node="extract_claims")
        valid_new_doc_ids = {s["doc_id"] for s in batch}
        seen_output_topics: set[str] = set()
        
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
    
                existing_by_topic[topic] = output_claims[-1]
                merged_count += 1
    
            else:
                output_claims.append(
                    Claim(
                        topic=topic,
                        source_doc_ids=new_source_doc_ids,
                        confidence=generated.confidence,
                    )
                )
    
                existing_by_topic[topic] = output_claims[-1]
                new_count += 1

    output_claims = list({c.topic: c for c in output_claims}.values())  
    logger.debug(
        "claims: %d new, %d merged",
        new_count,
        merged_count,
    )
    _emit(task_id, "claims_extracted", count=new_count, merged=merged_count, topics=[claim.topic for claim in output_claims])

    return {"claims": output_claims}