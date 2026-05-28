import asyncio
import hashlib
import os
import logging
from typing import Optional
import httpx
import trafilatura
from .state import Document, Query

logger = logging.getLogger(__name__)

SEARXNG_URL = os.environ["SEARXNG_INTERNAL_URL"]
SCRAPE_CONCURRENCY = int(os.environ.get("SCRAPE_CONCURRENCY", "6"))
SCRAPE_TIMEOUT = float(os.environ.get("SCRAPE_TIMEOUT", "10.0"))
MAX_CONTENT_CHARS = 8000  # ~2k tokens, plenty for the summarizer

search_http: httpx.AsyncClient | None = None


def _doc_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


async def search_searxng(query: str, max_results: int = 20) -> list[dict]:
    """Hit SearxNG's JSON API."""
    try:
        resp = await search_http.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general, science, it"},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results[:max_results]
    
    except Exception as e:
        logger.exception(f"SearxNG search failed for '{query}': {e}")
        return []


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


async def search_and_scrape_query(
    query: Query,
    seen_urls: set[str],
    max_results: int,
    target_docs: int,
) -> list[Document]:
    """Run one query, scrape new URLs only, return Documents.
    
    `seen_urls` is a snapshot of what's already been scraped across the run.
    URLs in that set are skipped entirely — no fetch, no extraction.""" # injected in main.py lifespan; shared across calls for efficiency
    hits = await search_searxng(query.query, max_results=max_results)
    
    # Filter out URLs we've already seen
    new_hits = [h for h in hits if h.get("url") and h["url"] not in seen_urls]
    if not new_hits:
        logger.info(f"query '{query.query}': all {len(hits)} hits already seen")
        return []
    
    # Scrape concurrently, bounded
    sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)
    scrape_tasks = [_scrape_one(h["url"], sem) for h in new_hits]
    contents = await asyncio.gather(*scrape_tasks, return_exceptions=False)
    
    good_docs: list[Document] = []
    for hit, content in zip(new_hits, contents):
        if not content or len(content) < 500:  # filter out failed scrapes and very short content
            continue  # drop docs where scrape failed or returned empty
        good_docs.append(Document(
            id=_doc_id(hit["url"]),
            url=hit["url"],
            title=hit.get("title", "")[:300],
            raw_content=content,
            source_query_id=query.id,
            search_score=float(hit.get("score", 0.0)),
        ))
    # even if output from searxng is already returned index by score, but removed those with no content so just in case, sort again by score.
    good_docs.sort(key=lambda d: d.search_score, reverse=True)
    docs = good_docs[:target_docs]
    
    logger.info(f"query '{query.query}': {len(hits)} hits, {len(good_docs)} good docs, {len(docs)} new docs scraped")
    return docs