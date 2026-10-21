"""Langfuse v3 tracing and optional capture of real node inputs.

Dataset input contains {node, state}; expected_output is deliberately unset.
Generated outputs live on the linked observation, not in the answer key.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from functools import wraps
from time import perf_counter

from langfuse import get_client
from pydantic import BaseModel

logger = logging.getLogger(__name__)
CAPTURE_DATASETS = os.getenv("RESEARCH_CAPTURE_DATASETS", "0").lower() in {"1", "true", "yes"}
ARM = os.getenv("RESEARCH_ARM", "current-mixed")
DATASET_PREFIX = os.getenv("RESEARCH_DATASET_PREFIX", "research").rstrip("-")
DATASETS = {
    "summarize_one_doc": f"{DATASET_PREFIX}-summaries",
    "extract_claims": f"{DATASET_PREFIX}-claims",
    "reflect": f"{DATASET_PREFIX}-reflection",
}
INPUT_KEYS = {
    "summarize_one_doc": ("original_query", "doc"),
    "extract_claims": ("original_query", "claims", "doc_summaries", "research_loop_count"),
    "reflect": (
        "original_query", "claims", "doc_summaries", "reflection_history",
        "search_queries", "reflected_doc_ids", "research_loop_count", "max_research_loops",
    ),
}
_ready_datasets: set[str] = set()


def jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


async def init_observability():
    """Create/upsert dataset names once at startup, not once per document."""
    if not CAPTURE_DATASETS:
        return
    for name in DATASETS.values():
        try:
            await asyncio.to_thread(
                get_client().create_dataset,
                name=name,
                description="Captured real research-node inputs; generated outputs are not ground truth.",
            )
            _ready_datasets.add(name)
        except Exception:
            logger.exception("Dataset setup failed for %s; capture disabled for this dataset", name)


async def _capture(payload, task_id, trace_id, observation_id, metadata):
    name = DATASETS[payload["node"]]
    if name not in _ready_datasets:
        return False
    # Repeat capture of the same invocation is idempotent. Separate research
    # tasks retain separate provenance, even when their inputs happen to match.
    identity = json.dumps([name, task_id, payload], sort_keys=True, ensure_ascii=False)
    item_id = hashlib.sha256(identity.encode()).hexdigest()
    try:
        await asyncio.to_thread(
            get_client().create_dataset_item,
            id=item_id,
            dataset_name=name,
            input=payload,
            metadata={**metadata, "source_task_id": task_id, "capture_schema_version": 1},
            source_trace_id=trace_id,
            source_observation_id=observation_id,
        )
        return True
    except Exception:
        logger.exception("Dataset capture failed: node=%s task=%s", payload["node"], task_id)
        return False


def capture_node(node_name):
    """Wrap the real node; capture successes and errors without copying logic."""
    def decorate(fn):
        @wraps(fn)
        async def wrapped(state, *args, **kwargs):
            lf = get_client()
            # Serialize BEFORE execution so mutations cannot change the case.
            payload = {"node": node_name, "state": jsonable({
                key: state[key] for key in INPUT_KEYS[node_name] if key in state
            })}
            metadata = {
                "arm": ARM, "node": node_name,
                "research_loop": state.get("research_loop_count", 0),
            }
            with lf.start_as_current_span(name=f"research-{node_name}", input=payload) as span:
                trace_id = lf.get_current_trace_id()
                observation_id = lf.get_current_observation_id()
                started = perf_counter()
                try:
                    result = await fn(state, *args, **kwargs)
                    span.update(output=jsonable(result))
                except asyncio.CancelledError:
                    metadata["status"] = "cancelled"
                    raise
                except Exception as exc:
                    metadata["status"] = "failed"
                    span.update(level="ERROR", status_message=str(exc))
                    raise
                else:
                    metadata["status"] = "complete"
                    return result
                finally:
                    metadata["node_latency_ms"] = round((perf_counter() - started) * 1000, 3)
                    span.update(metadata=metadata)
                    if CAPTURE_DATASETS and metadata["status"] != "cancelled":
                        capture_start = perf_counter()
                        captured = await _capture(
                            payload, state["task_id"], trace_id, observation_id, metadata,
                        )
                        span.update(metadata={
                            **metadata, "dataset_captured": captured,
                            "dataset_capture_ms": round((perf_counter() - capture_start) * 1000, 3),
                        })
        return wrapped
    return decorate
