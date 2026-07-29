#!/usr/bin/env python3
"""Turn run event-logs into an A/B comparison table.

    python analyze_events.py --arm A runs/armA/*.json --arm B runs/armB/*.json

Reads the JSON your /events endpoint returns ({"events": [...], "report": ...})
and reports, per arm: stage wall-clock, summarizer throughput/quality, and
report-level counts. Medians + IQR, because latency distributions are skewed
and a single slow scrape run will wreck a mean.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# stage-start event -> the event that marks its completion
STAGE_END = {
    "generating_queries": "queries_generated",
    "scraping": "scrape_complete",
    "extracting_claims": "claims_extracted",
    "reflecting": "reflection",
    "planning": "plan_ready",
    "stitching": "report_ready",
}


def stage_durations(events: list[dict]) -> dict[str, float]:
    """Wall-clock per stage. Summarize/write have no stage event of their own —
    they're fan-outs — so they're bracketed by the surrounding markers."""
    out: dict[str, float] = {}
    open_stage: tuple[str, float] | None = None

    for e in events:
        if e["type"] == "stage":
            open_stage = (e["stage"], e["ts"])
        elif open_stage and e["type"] == STAGE_END.get(open_stage[0]):
            out[open_stage[0]] = round(e["ts"] - open_stage[1], 1)
            open_stage = None

    def ts_of(pred) -> float | None:
        return next((e["ts"] for e in events if pred(e)), None)

    def last_ts_of(pred) -> float | None:
        return next((e["ts"] for e in reversed(events) if pred(e)), None)

    scrape_done = ts_of(lambda e: e["type"] == "scrape_complete")
    last_summary = last_ts_of(lambda e: e["type"] == "doc_summarized")
    if scrape_done is not None and last_summary is not None:
        out["summarize_fanout"] = round(last_summary - scrape_done, 1)

    first_write = ts_of(lambda e: e["type"] == "writing_section")
    last_write = last_ts_of(lambda e: e["type"] == "section_written")
    if first_write is not None and last_write is not None:
        out["write_sections"] = round(last_write - first_write, 1)

    out["total"] = round(events[-1]["ts"], 1)
    return out


def summarizer_metrics(events: list[dict]) -> dict[str, float]:
    docs = [e for e in events if e["type"] == "doc_summarized"]
    if not docs:
        return {}
    rel = [e for e in docs if e.get("relevant")]
    findings = [f for e in rel for f in e.get("key_findings", [])]

    m: dict[str, float] = {
        "docs_summarized": len(docs),
        "relevance_precision": round(len(rel) / len(docs), 3),
    }
    if rel:
        counts = [len(e.get("key_findings", [])) for e in rel]
        m["findings_per_doc_med"] = st.median(counts)
        # Distinct list-lengths the model actually produced. 1-2 means it has
        # latched onto a fixed count regardless of what the document contains.
        m["findings_count_variety"] = len(set(counts))
        # overview is Field(default="") -> OPTIONAL in the emitted JSON schema,
        # so guided decoding is free to skip it entirely. Fill rate catches that.
        have_ov = [e for e in rel if "overview" in e]
        if have_ov:
            m["overview_fill_rate"] = round(
                sum(1 for e in have_ov if (e.get("overview") or "").strip()) / len(have_ov), 3
            )
    if findings:
        m["findings_total"] = len(findings)
        m["pct_with_number"] = round(
            sum(1 for f in findings if re.search(r"\d", f)) / len(findings), 3
        )
        # Proxy for max_length truncation: guided decoding cuts mid-sentence,
        # a validation retry gets a chance to rewrite shorter.
        m["pct_unterminated"] = round(
            sum(1 for f in findings if not f.strip().endswith((".", "!", "?", '"', ")")))
            / len(findings), 3
        )
    # Emitted by the instrumented validated.py; absent on guided arms.
    attempts = [e["attempts"] for e in docs if "attempts" in e]
    if attempts:
        m["first_attempt_ok"] = round(sum(1 for a in attempts if a == 1) / len(attempts), 3)
        m["mean_attempts"] = round(st.mean(attempts), 2)
        m["fallback_rate"] = round(
            sum(1 for e in docs if e.get("fallback_used")) / len(docs), 3
        )
    return m


def report_metrics(payload: dict, events: list[dict]) -> dict[str, float]:
    m: dict[str, float] = {}
    claims = next((e for e in events if e["type"] == "claims_extracted"), None)
    if claims:
        m["claims"] = claims.get("count", 0) + claims.get("merged", 0)
    secs = [e for e in events if e["type"] == "section_written"]
    if secs:
        m["sections"] = len(secs)
        m["citations_med"] = st.median([e.get("citations", 0) for e in secs])
    rpt = payload.get("report") or ""
    if rpt:
        m["report_chars"] = len(rpt)
        m["refs"] = len(re.findall(r"^\d+\. \[", rpt, re.M))
    return m


def load(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(path.read_text())
    return payload["events"], payload


def summarize_arm(paths: list[Path]) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = defaultdict(list)
    for p in paths:
        events, payload = load(p)
        if not events:
            continue
        for src in (stage_durations(events),
                    summarizer_metrics(events),
                    report_metrics(payload, events)):
            for k, v in src.items():
                acc[k].append(v)
    return acc


def fmt(vals: list[float]) -> str:
    if not vals:
        return "-"
    med = st.median(vals)
    if len(vals) < 3:
        return f"{med:g}"
    lo, hi = min(vals), max(vals)
    return f"{med:g} [{lo:g}-{hi:g}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", nargs="+", metavar=("NAME", "FILE"),
                    required=True, help="--arm A runs/a/*.json  (repeatable)")
    args = ap.parse_args()

    arms = {a[0]: summarize_arm([Path(f) for f in a[1:]]) for a in args.arm}
    names = list(arms)
    keys: list[str] = []
    for a in arms.values():
        for k in a:
            if k not in keys:
                keys.append(k)

    w = max(len(k) for k in keys) + 2
    print(f"{'metric':<{w}}" + "".join(f"{n:>22}" for n in names))
    print("-" * (w + 22 * len(names)))
    for k in keys:
        row = f"{k:<{w}}"
        for n in names:
            row += f"{fmt(arms[n].get(k, [])):>22}"
        print(row)
    print(f"\nn runs per arm: " + ", ".join(f"{n}={len(arms[n].get('total', []))}" for n in names))


if __name__ == "__main__":
    main()