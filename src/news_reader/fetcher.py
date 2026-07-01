import hashlib
from datetime import datetime, timezone

import feedparser
import httpx
from trafilatura import extract

from news_reader.sources import Source


async def fetch_rss(source: Source, client: httpx.AsyncClient) -> list[dict]:
    url = source.feed_url or source.url
    resp = await client.get(url, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    articles = []
    for entry in feed.entries:
        link = entry.get("link", "")
        content_hash = hashlib.sha256(link.encode()).hexdigest()

        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()

        tags = []
        if hasattr(entry, "tags"):
            tags = [t.get("term", "") for t in entry.tags if t.get("term")]

        articles.append({
            "source_id": source.id,
            "title": entry.get("title", ""),
            "author": entry.get("author"),
            "summary": entry.get("summary"),
            "link": link,
            "content_hash": content_hash,
            "keywords": ",".join(tags) if tags else None,
            "published_at": published,
        })
    return articles


async def fetch_scrape(source: Source, client: httpx.AsyncClient) -> list[dict]:
    # For scrape sources: fetch the HTML, then use trafilatura to extract article content
    resp = await client.get(source.url, timeout=30)
    resp.raise_for_status()

    text = extract(resp.text)
    if not text:
        return []

    content_hash = hashlib.sha256(source.url.encode()).hexdigest()
    return [{
        "source_id": source.id,
        "title": "",
        "author": None,
        "summary": text[:500],
        "link": source.url,
        "content_hash": content_hash,
        "keywords": None,
        "published_at": None,
    }]
