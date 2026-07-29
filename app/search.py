import asyncio
import hashlib
import os
import re
import logging
import re
from urllib.parse import urlparse 
from typing import Iterable, Optional
import httpx
import trafilatura
from .state import Document, Query

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ["SEARXNG_INTERNAL_URL"]
SCRAPE_CONCURRENCY = int(os.environ.get("SCRAPE_CONCURRENCY", "6"))
SCRAPE_TIMEOUT = float(os.environ.get("SCRAPE_TIMEOUT", "10.0"))
MAX_CONTENT_CHARS = 8000  # ~2k tokens, plenty for the summarizer

search_http: httpx.AsyncClient | None = None

# Binary / non-HTML extensions trafilatura can't extract.
_SKIP_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".ogg", ".webm", ".m4a",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".exe", ".dmg", ".iso",
)
def _is_scrapeable(url: str) -> bool:
    """Trafilatura can't extract binaries. Searxng already handles
    domain-level blocking via hostnames.remove in settings.yml."""
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return False
    return not path.endswith(_SKIP_EXTENSIONS)


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


async def search_searxng(query: str, categories: str = "general", max_results: int = 20) -> list[dict]:
    """Hit SearxNG's JSON API."""
    try:
        resp = await search_http.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": categories},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results[:max_results]
    
    except Exception as e:
        logger.exception(f"SearxNG search failed for '{query}': {e}")
        return []


async def search_query(
    query: Query,
    seen_urls: set[str],
    max_results: int,
) -> list[Document]:
    """Searxng only — no scraping. Returns hit metadata for the
    pre_scrape node to triage."""
    hits = await search_searxng(query.query, categories=query.category, max_results=max_results)

    out: list[Document] = []
    for h in hits:
        url = h.get("url")
        if not url or url in seen_urls:
            continue
        if not _is_scrapeable(url):
            continue
        out.append(Document(
            id=_doc_id(url),
            url=url,
            title=(h.get("title") or ""),
            content=(h.get("content") or ""), # searxng content
            source_query_id=query.id,
            search_score=float(h.get("score", 0.0)),
            category=query.category,
        ))
    logger.info(f"query '{query.query}': {len(hits)} raw hits, {len(out)} scrapeable new hits")
    return out

async def _scrape_one(
    url: str,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    """Fetch + trafilatura extract under a concurrency limit."""
    async with sem:
        try:
            resp = await search_http.get(url, timeout=SCRAPE_TIMEOUT, follow_redirects=True)
            resp.raise_for_status()
            extracted = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if not extracted:
                return None
            return extracted[:MAX_CONTENT_CHARS]
        except Exception as e:
            logger.warning(f"scrape failed for {url}: {e}")
            return None

async def scrape_hits(hits: list[Document]) -> list[Document]:
    """Fetch + extract for an already-filtered hit list."""
    sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)
    tasks = [_scrape_one(h.url, sem) for h in hits]
    contents = await asyncio.gather(*tasks, return_exceptions=False)

    docs: list[Document] = []
    for hit, content in zip(hits, contents):
        if not content or len(content) < 500:
            continue
        docs.append(Document(
            id=hit.id,
            url=hit.url,
            title=hit.title,
            content=content,
            source_query_id=hit.source_query_id,
            search_score=hit.search_score,
        ))
    docs.sort(key=lambda d: d.search_score, reverse=True)
    logger.info(f"scrape: {len(hits)} candidates → {len(docs)} good docs")
    return docs

# Matches [d_xxx] or [d_xxx, d_yyy, d_zzz] with flexible whitespace.
# Citations now point directly at doc_ids — the claim layer is purely an
_CITE_RE = re.compile(
    r"\[(d_[a-f0-9]{12}(?:\s*,\s*d_[a-f0-9]{12})*)\]"
)
 
 
def _strip_d(tok: str) -> str:
    """'d_fb2843aa4730' -> 'fb2843aa4730'. Defensive against accidental whitespace."""
    t = tok.strip()
    return t[2:] if t.startswith("d_") else t
 
 
def collect_ordered_doc_ids(ordered_sections) -> list[str]:
    """Doc-ids in order of first appearance across the report — standard
    academic numbering, reads better than sorted-by-id."""
    seen: set[str] = set()
    order: list[str] = []
    for sec in ordered_sections:
        for match in _CITE_RE.finditer(sec.body_markdown):
            for tok in match.group(1).split(","):
                did = _strip_d(tok)
                if did and did not in seen:
                    seen.add(did)
                    order.append(did)
    return order
 
# Collapse runs of 2+ adjacent numeric citation groups: [8, 9][8, 9, 10, 11] -> [8, 9, 10, 11]
_NUMCITE_RUN = re.compile(r"(?:\[\d+(?:\s*,\s*\d+)*\]\s*){2,}")
 
def merge_adjacent_numeric_cites(text: str) -> str:
    def merge(m: re.Match) -> str:
        nums = sorted({int(n) for n in re.findall(r"\d+", m.group(0))})
        return "[" + ", ".join(map(str, nums)) + "]"
    return _NUMCITE_RUN.sub(merge, text)

def rewrite_citations(body: str, doc_to_ref) -> str:
    """Replace [d_xxx] / [d_xxx, d_yyy] with [1] / [1, 3], dedup + sort.
    Doc-ids that don't resolve are dropped silently rather than left as raw markers."""
    def replace(m: re.Match) -> str:
        refs: list[int] = []
        for tok in m.group(1).split(","):
            did = _strip_d(tok)
            n = doc_to_ref.get(did)
            if n is not None and n not in refs:
                refs.append(n)
        if not refs:
            return ""
        refs.sort()
        return "[" + ", ".join(str(r) for r in refs) + "]"
    return _CITE_RE.sub(replace, body)
