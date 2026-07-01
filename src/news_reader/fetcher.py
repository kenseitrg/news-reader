import hashlib
import json
from datetime import datetime, timezone

import feedparser
import httpx
from bs4 import BeautifulSoup

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
    name_lower = source.name.lower()

    if "hacker news" in name_lower:
        ts_24h = int(datetime.now(timezone.utc).timestamp()) - 86400
        api_url = f"https://hn.algolia.com/api/v1/search?tags=story&numericFilters=created_at_i%3E{ts_24h}&hitsPerPage=50"
        resp = await client.get(api_url, timeout=30)
        resp.raise_for_status()
        return _parse_hacker_news(resp.text, source.id)

    resp = await client.get(
        source.url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    resp.raise_for_status()

    if "dtf" in name_lower:
        return _parse_dtf(resp.text, source.id, source.url)
    return []


def _parse_hacker_news(body: str, source_id: int) -> list[dict]:
    """Parse HN Algolia API response JSON."""
    data = json.loads(body)
    articles = []
    for hit in data.get("hits", []):
        title = hit.get("title", "")
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
        points = hit.get("points", 0)
        author = hit.get("author", "")
        content_hash = hashlib.sha256(url.encode()).hexdigest()
        created = hit.get("created_at")

        articles.append({
            "source_id": source_id,
            "title": title,
            "author": author,
            "summary": f"Score: {points} points | {hit.get('num_comments', 0)} comments",
            "link": url,
            "content_hash": content_hash,
            "keywords": None,
            "published_at": created,
        })

    articles.sort(key=lambda a: int(a["summary"].split()[1]), reverse=True)
    return articles


def _parse_dtf(html: str, source_id: int, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    # Detect section from URL
    section = "/hard/" if "/hard" in url else "/cinema/"

    for item in soup.select(".content-list > div.content--short"):
        links = {}
        for a in item.find_all("a"):
            text = a.get_text(strip=True)
            href = a.get("href", "")
            links[href] = text

        # Find the actual article link (not a comment link, not a time link)
        article_link = None
        title = None
        for href, text in links.items():
            if section in href and "#comments" not in href and text:
                if not article_link or len(text) > len(links.get(article_link, "")):
                    article_link = href
                    title = text

        if not article_link:
            continue

        # Also skip items where the longest text is very short (just a time)
        if title and len(title) < 10:
            continue

        full_link = f"https://dtf.ru{article_link}" if article_link.startswith("/") else article_link
        content_hash = hashlib.sha256(full_link.encode()).hexdigest()

        author_el = item.select_one("[class*=author__name]")
        author = author_el.get_text(strip=True) if author_el else None

        time_el = item.find("time")
        published = time_el.get("datetime") if time_el else None

        summary_el = item.select_one(".block-text")
        summary = summary_el.get_text(strip=True)[:500] if summary_el else None

        articles.append({
            "source_id": source_id,
            "title": title,
            "author": author,
            "summary": summary,
            "link": full_link,
            "content_hash": content_hash,
            "keywords": None,
            "published_at": published,
        })

    return articles
