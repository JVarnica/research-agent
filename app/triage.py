## no good to run got code from other sections.


"""
Pre-scrape triage: decide which search hits are worth scraping.

Three interchangeable arms so you can A/B on the same question:

    TRIAGE_MODE=rerank   cross-encoder reranker  (recommended)
    TRIAGE_MODE=llm      single batched LLM call (fixed version of the old one)
    TRIAGE_MODE=off      keep top-N by search_score, no model at all

Set via env var or pass `mode=` to pre_scrape().

Design notes
------------
* The reranker never touches your vLLM queue. It is a ~570M cross-encoder that
  scores all hits in one forward pass, typically <1s for 40 hits, and can run on
  CPU if you'd rather keep both GPUs for generation.
* Triage is an OPTIMISATION, not a correctness gate. Every failure path here
  fails OPEN (keep the hits) rather than closed. The old code returned an empty
  set on error, which silently dropped 6 hits per failed chunk and made the
  outer top-N fallback unreachable.
* Scraping is cheap (~0.2s/URL). The point of triage is to cut the number of
  *summarisation* calls, so it must itself cost near zero.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

# Hard ceiling on hits sent to the scraper, whatever the mode.
MAX_SCRAPE = 20

# Reranker choice. bge-reranker-v2-m3 is multilingual and strong; the MiniLM
# option is ~25x smaller and English-only if you want it on CPU next to the
# chat users. Both are cross-encoders — they take (query, passage) pairs.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 22M, English, CPU-fast

# Device for the reranker. "auto" picks cuda -> mps -> cpu.
# Consider pinning to "cpu" in production so it never contends with vLLM.
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")

# Minimum sigmoid-normalised relevance to survive triage. This is model-specific
# and needs calibrating against YOUR queries — start permissive and rely on
# top_k, then tighten once you've eyeballed a few runs. Set to 0.0 to disable.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.0"))

TRIAGE_MODE = os.getenv("TRIAGE_MODE", "rerank")


# --------------------------------------------------------------------------
# Arm 1: cross-encoder reranker
# --------------------------------------------------------------------------

_reranker = None
_reranker_lock = asyncio.Lock()


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_reranker():
    """Blocking model load. Called once, off the event loop."""
    from sentence_transformers import CrossEncoder

    device = _resolve_device(RERANKER_DEVICE)
    logger.info("loading reranker %s on %s", RERANKER_MODEL, device)
    return CrossEncoder(RERANKER_MODEL, device=device, max_length=512)


async def _get_reranker():
    """Lazy singleton. The lock stops N concurrent callers loading N copies."""
    global _reranker
    if _reranker is None:
        async with _reranker_lock:
            if _reranker is None:
                _reranker = await asyncio.to_thread(_load_reranker)
    return _reranker


def _hit_text(hit: Any) -> str:
    """What the reranker actually reads. Title carries most of the signal;
    the URL path is surprisingly informative (/blog/ vs /docs/ vs /definition/)."""
    title = (getattr(hit, "title", "") or "").strip()
    snippet = (getattr(hit, "snippet", "") or "").strip()
    url = (getattr(hit, "url", "") or "").strip()
    return f"{title}\n{url}\n{snippet}"[:1200]


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


async def rerank_hits(
    question: str,
    hits: Sequence[Any],
    top_k: int = MAX_SCRAPE,
    min_score: float = RERANK_MIN_SCORE,
) -> list[Any]:
    """Score every hit against the question, return the best `top_k`.

    Returns hits in descending relevance order. Raises nothing — on any
    failure the caller's fail-open path handles it.
    """
    if not hits:
        return []

    model = await _get_reranker()
    pairs = [(question, _hit_text(h)) for h in hits]

    # predict() is CPU/GPU-bound and releases the GIL inside torch, but wrap it
    # anyway so a 40-pair batch can't stall the event loop and your SSE stream.
    raw = await asyncio.to_thread(model.predict, pairs, batch_size=32)

    scored = []
    for hit, s in zip(hits, raw):
        score = _sigmoid(float(s))
        scored.append((score, hit))
        # Attach for logging/debug; harmless if the model has no such field.
        try:
            object.__setattr__(hit, "rerank_score", score)
        except Exception:
            pass

    scored.sort(key=lambda t: t[0], reverse=True)

    kept = [h for score, h in scored if score >= min_score][:top_k]

    if logger.isEnabledFor(logging.DEBUG):
        for score, h in scored:
            logger.debug(
                "%.3f %s %s",
                score,
                "KEEP" if h in kept else "drop",
                (getattr(h, "title", "") or "")[:70],
            )
    return kept


# --------------------------------------------------------------------------
# Arm 2: single batched LLM call
# --------------------------------------------------------------------------
# Fixed relative to the original:
#   - ONE call, not len(hits)/6 calls
#   - no maxItems bound (that was the expensive schema feature)
#   - verdict carries id + keep only; no free-text `reason` to decode
#   - max_tokens cap as the real anti-rambling lever
#   - thinking disabled

class HitVerdict(BaseModel):
    id: str
    keep: bool


class TriageVerdicts(BaseModel):
    verdicts: list[HitVerdict] = Field(
        description="Exactly one verdict per hit, same ids as the input."
    )


TRIAGE_PROMPT = """You are triaging search results before scraping.

Research question: {question}

Return one verdict per hit. Default keep=false. Set keep=true ONLY if the title,
URL, or snippet names something SPECIFIC to the research question: a relevant
person, place, event, date, or clearly on-topic discussion. A hit that merely
shares a common word with the question in an unrelated context is NOT relevant.

Always keep=false for: dictionary/thesaurus/vocabulary pages; unrelated software
docs; unrelated medical pages; OS or product help pages; generic hub, category,
or vendor landing pages; SEO listicles; video/forum/Q&A links whose snippet
shows no substantive on-topic content.

You are seeing all hits at once — if several are near-duplicates of each other,
keep only the best one.

When unsure, drop it.

Hits:
{hits_json}"""


async def llm_triage(question: str, hits: Sequence[Any], get_clients) -> list[Any]:
    """Single-call LLM triage. Kept as a comparison arm."""
    if not hits:
        return []

    hits_json = json.dumps(
        [
            {
                "id": h.id,
                "title": h.title,
                "url": h.url,
                "snippet": (h.snippet or "")[:300],
            }
            for h in hits
        ],
        ensure_ascii=False,
    )

    structured = get_clients().struct_cheap_llm(TriageVerdicts)
    result: TriageVerdicts = await structured.ainvoke(
        [
            {
                "role": "user",
                "content": TRIAGE_PROMPT.format(question=question, hits_json=hits_json),
            }
        ],
        # Both of these matter more than any schema bound:
        max_tokens=12 * len(hits) + 64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    valid = {h.id for h in hits}
    keep_ids = {v.id for v in result.verdicts if v.keep and v.id in valid}
    return [h for h in hits if h.id in keep_ids]


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------


async def pre_scrape(
    state: dict,
    *,
    mode: Literal["rerank", "llm", "off"] | None = None,
    get_clients=None,
    emit=None,
) -> dict:
    """Triage node. Drop-in replacement for the old pre_scrape.

    Wire it as:  pre_scrape(state, get_clients=get_clients, emit=_emit)
    or import get_clients/_emit directly and drop the kwargs.
    """
    task_id = state["task_id"]
    hits = state.get("search_hits", []) or []
    mode = mode or TRIAGE_MODE

    def _e(event: str, **data):
        if emit:
            emit(task_id, event, **data)

    if not hits:
        _e("triage_complete", before=0, after=0, mode=mode)
        return {"hits_to_scrape": []}

    _e("stage", stage="triaging", message=f"Triaging {len(hits)} hits ({mode})")

    loop = asyncio.get_running_loop()
    t0 = loop.time()

    try:
        if mode == "rerank":
            kept = await rerank_hits(state["original_query"], hits, top_k=MAX_SCRAPE)
        elif mode == "llm":
            if get_clients is None:
                raise ValueError("llm mode needs get_clients")
            kept = await llm_triage(state["original_query"], hits, get_clients)
        else:  # "off"
            kept = list(hits)
    except Exception as e:
        # Fail OPEN. Triage is an optimisation; losing it must not lose the run.
        logger.exception("triage failed (%s), falling back to search_score: %s", mode, e)
        kept = sorted(hits, key=lambda h: getattr(h, "search_score", 0.0), reverse=True)

    # Single truncation, single emit.
    if len(kept) > MAX_SCRAPE:
        kept = sorted(
            kept, key=lambda h: getattr(h, "search_score", 0.0), reverse=True
        )[:MAX_SCRAPE]

    elapsed = loop.time() - t0
    logger.info(
        "pre_scrape[%s]: %d hits -> %d kept in %.2fs", mode, len(hits), len(kept), elapsed
    )
    _e(
        "triage_complete",
        before=len(hits),
        after=len(kept),
        mode=mode,
        elapsed=round(elapsed, 2),
    )
    return {"hits_to_scrape": kept}