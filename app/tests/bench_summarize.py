#!/usr/bin/env python3
"""Paired micro-benchmark: guided decoding vs validate-and-retry.

This is where the real signal lives. The end-to-end pipeline has too many
moving parts (search results drift, scrape failures, loop counts) to measure
a decoding change against. Here the corpus is frozen, so the ONLY thing that
differs between arms is how the model is asked to produce JSON.

Usage:
    python -m app.bench_summarize --docs frozen_docs.json --concurrency 16
    python -m app.bench_summarize --docs frozen_docs.json --concurrency 1 --repeats 3

Capture a frozen corpus once from a real run, e.g. at the end of scrape_node:
    Path("frozen_docs.json").write_text(
        json.dumps([d.model_dump() for d in docs], ensure_ascii=False))

Design notes:
  * PAIRED — every document goes through both arms, so per-document difficulty
    cancels out. Use a signed-rank test, not an unpaired t-test.
  * INTERLEAVED — arms alternate per repeat rather than running all-A-then-all-B,
    so GPU thermal state and vLLM cache warmth hit both arms equally.
  * WARMUP — the first guided call of a process pays grammar compilation, and
    the first call of any kind pays model warm-up. Both are discarded.
  * Run at concurrency=1 AND at your real fan-out width. Per-request overhead
    and queueing behaviour are different effects and they can point opposite ways.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics as st
import time
from pathlib import Path

from pydantic import ValidationError

from ..state import Document, DocSummaryLLM
from ..llm import init_clients, get_clients
from ..nodes import SUMMARIZE_PROMPT
from ..validate import ainvoke_validated

ARMS = ("guided", "validated")


def build_messages(doc: Document, question: str) -> list[dict]:
    return [{"role": "user", "content": SUMMARIZE_PROMPT.format(
        question=question, title=doc.title, url=doc.url, content=doc.snippet,
    )}]


async def run_one(arm: str, doc: Document, question: str, sem: asyncio.Semaphore) -> dict:
    msgs = build_messages(doc, question)
    rec: dict = {"arm": arm, "doc_id": doc.id, "ok": False,
                 "attempts": 1, "fallback": False}
    async with sem:
        t0 = time.perf_counter()
        try:
            if arm == "guided":
                llm = get_clients().struct_cheap_llm(DocSummaryLLM)
                res: DocSummaryLLM = await llm.ainvoke(msgs)
            else:
                stats: dict = {}
                res = await ainvoke_validated(
                    get_clients().cheap_llm(), DocSummaryLLM, msgs,
                    max_attempts=3, guided_fallback=None, stats=stats,
                )
                rec["attempts"] = stats.get("attempts", 1)
                rec["fallback"] = stats.get("fallback_used", False)
            rec["ok"] = True
        except ValidationError as e:
            rec["error"] = f"validation:{len(e.errors())}"
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}"
        rec["latency"] = round(time.perf_counter() - t0, 3)

    if rec["ok"]:
        rec.update(quality(res))
    return rec


def quality(s: DocSummaryLLM) -> dict:
    f = s.key_findings
    return {
        "relevant": s.relevant,
        "n_findings": len(f),
        "n_quotes": len(s.quotes),
        "overview_filled": bool(s.overview.strip()),
        "overview_len": len(s.overview),
        # Specificity: the prompt demands a date, number, or named entity.
        "pct_with_number": (sum(1 for x in f if re.search(r"\d", x)) / len(f)) if f else 0.0,
        # Truncation proxy — a field cut off by max_length rarely ends cleanly.
        "n_unterminated": sum(1 for x in f if not x.strip().endswith((".", "!", "?", '"', ")"))),
    }


async def warmup(question: str, doc: Document) -> None:
    """Discarded. Pays model load + per-schema grammar compilation once, so
    neither cost lands inside a measured arm."""
    sem = asyncio.Semaphore(1)
    for arm in ARMS:
        await run_one(arm, doc, question, sem)


async def main_async(args: argparse.Namespace) -> None:
    init_clients()
    docs = [Document(**d) for d in json.loads(Path(args.docs).read_text())]
    if args.limit:
        docs = docs[: args.limit]
    print(f"{len(docs)} docs | concurrency={args.concurrency} | repeats={args.repeats}")

    await warmup(args.question, docs[0])
    print("warmup done\n")

    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []
    rng = random.Random(args.seed)

    for rep in range(args.repeats):
        # Alternate which arm goes first so ordering can't favour one of them.
        order = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            shuffled = docs[:]
            rng.shuffle(shuffled)  # kill positional effects in the queue
            t0 = time.perf_counter()
            res = await asyncio.gather(*[
                run_one(arm, d, args.question, sem) for d in shuffled
            ])
            wall = time.perf_counter() - t0
            for r in res:
                r["repeat"] = rep
            rows.extend(res)
            print(f"  rep{rep} {arm:<10} wall={wall:6.1f}s  ok={sum(r['ok'] for r in res)}/{len(res)}")

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nraw -> {args.out}\n")
    report(rows)


def report(rows: list[dict]) -> None:
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}

    def agg(rs: list[dict], key: str, only_ok: bool = True) -> list[float]:
        return [r[key] for r in rs if key in r and (r["ok"] or not only_ok)]

    print(f"{'metric':<24}" + "".join(f"{a:>14}" for a in ARMS) + "     delta")
    print("-" * 68)

    def line(name: str, fn, pct: bool = False) -> None:
        vals = [fn(by_arm[a]) for a in ARMS]
        if any(v is None for v in vals):
            return
        fmtv = (lambda v: f"{v*100:.1f}%") if pct else (lambda v: f"{v:.3f}")
        d = vals[1] - vals[0]
        ds = f"{d*100:+.1f}pp" if pct else f"{d:+.3f}"
        print(f"{name:<24}" + "".join(f"{fmtv(v):>14}" for v in vals) + f"{ds:>10}")

    def med(k):
        return lambda rs: st.median(agg(rs, k)) if agg(rs, k) else None

    def mean(k):
        return lambda rs: st.mean(agg(rs, k)) if agg(rs, k) else None

    def p95(k):
        def f(rs):
            v = sorted(agg(rs, k))
            return v[int(len(v) * 0.95)] if v else None
        return f

    line("latency p50 (s)", med("latency"))
    line("latency p95 (s)", p95("latency"))
    line("success rate", lambda rs: sum(r["ok"] for r in rs) / len(rs), pct=True)
    line("findings/doc", mean("n_findings"))
    line("quotes/doc", mean("n_quotes"))
    line("overview fill", mean("overview_filled"), pct=True)
    line("overview len", mean("overview_len"))
    line("relevance rate", mean("relevant"), pct=True)
    line("specificity(num)", mean("pct_with_number"), pct=True)
    line("unterminated/doc", mean("n_unterminated"))
    line("mean attempts", mean("attempts"))

    # Degenerate-output check: how many distinct findings-counts each arm
    # produced. Collapsing to one or two values means the model is emitting a
    # fixed shape regardless of what the document actually contains.
    for a in ARMS:
        counts = agg(by_arm[a], "n_findings")
        if counts:
            hist = {c: counts.count(c) for c in sorted(set(counts))}
            print(f"\n{a:<10} findings-count histogram: {hist}")

    # Paired per-document latency comparison — the number that actually matters.
    pairs: dict[tuple, dict] = {}
    for r in rows:
        if r["ok"]:
            pairs.setdefault((r["doc_id"], r["repeat"]), {})[r["arm"]] = r["latency"]
    both = [(v["guided"], v["validated"]) for v in pairs.values() if len(v) == 2]
    if both:
        diffs = [b - a for a, b in both]
        wins = sum(1 for d in diffs if d < 0)
        print(f"\npaired n={len(both)} | validated faster on {wins} "
              f"({wins/len(both)*100:.0f}%) | median diff {st.median(diffs):+.3f}s")
        print("  -> feed `diffs` to scipy.stats.wilcoxon for a p-value")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="JSON list of Document dicts")
    ap.add_argument("--question", default="Why is Europe stagnating economically?")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_raw.json")
    asyncio.run(main_async(ap.parse_args()))