import hashlib
import json
from datetime import datetime, timezone

import feedparser
import httpx
from bs4 import BeautifulSoup
from trafilatura import extract

from news_reader.sources import Source


async def fetch_rss(source: Source, client: httpx.AsyncClient) -> list[dict]:
    url = source.feed_url or source.url
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = await client.get(url, timeout=30, headers=headers, follow_redirects=True)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    articles = []
    for entry in feed.entries:
        link = entry.get("link", "")
        content_hash = hashlib.sha256(link.encode()).hexdigest()

        published = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(
                *entry.published_parsed[:6], tzinfo=timezone.utc
            ).isoformat()

        tags = []
        if hasattr(entry, "tags"):
            tags = [t.get("term", "") for t in entry.tags if t.get("term")]

        articles.append(
            {
                "source_id": source.id,
                "title": entry.get("title", ""),
                "author": entry.get("author"),
                "summary": entry.get("summary"),
                "link": link,
                "content_hash": content_hash,
                "keywords": ",".join(tags) if tags else None,
                "published_at": published,
            }
        )
    return articles


async def fetch_scrape(source: Source, client: httpx.AsyncClient) -> list[dict]:
    name_lower = source.name.lower()

    if "hacker news" in name_lower:
        ts_24h = int(datetime.now(timezone.utc).timestamp()) - 86400
        api_url = f"https://hn.algolia.com/api/v1/search?tags=story&numericFilters=created_at_i%3E{ts_24h}&hitsPerPage=50"
        resp = await client.get(api_url, timeout=30)
        resp.raise_for_status()
        assert source.id is not None
        return _parse_hacker_news(resp.text, source.id)

    resp = await client.get(
        source.url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    resp.raise_for_status()

    if "dtf" in name_lower:
        assert source.id is not None
        return _parse_dtf(resp.text, source.id, source.url)
    return []


def _parse_hacker_news(body: str, source_id: int) -> list[dict]:
    """Parse HN Algolia API response JSON."""
    data = json.loads(body)
    hits = sorted(data.get("hits", []), key=lambda h: h.get("points", 0), reverse=True)
    articles = []
    for hit in hits:
        title = hit.get("title", "")
        url = (
            hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
        )
        author = hit.get("author", "")
        content_hash = hashlib.sha256(url.encode()).hexdigest()
        created = hit.get("created_at")

        articles.append(
            {
                "source_id": source_id,
                "title": title,
                "author": author,
                "summary": None,
                "link": url,
                "content_hash": content_hash,
                "keywords": None,
                "published_at": created,
            }
        )

    return articles


async def fetch_article_content(url: str, client: httpx.AsyncClient) -> str | None:
    """Download a single article page and extract readable text."""
    try:
        resp = await client.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            },
        )
        resp.raise_for_status()
        text = extract(resp.text)
        return text.strip() if text else None
    except Exception:
        return None


def _parse_dtf(html: str, source_id: int, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    # Detect section from URL
    section = "/hard/" if "/hard" in url else "/cinema/"

    for item in soup.select(".content-list > div.content--short"):
        links: dict[str, str] = {}
        for a in item.find_all("a"):
            text = a.get_text(strip=True)
            href = a.get("href")
            if isinstance(href, str):
                links[href] = text

        article_link: str | None = None
        title: str | None = None
        for href, text in links.items():
            if section in href and "#comments" not in href and text:
                if not article_link or len(text) > len(links.get(article_link, "")):
                    article_link = href
                    title = text

        if not article_link:
            continue

        if title and len(title) < 10:
            continue

        full_link = (
            f"https://dtf.ru{article_link}"
            if article_link.startswith("/")
            else article_link
        )
        content_hash = hashlib.sha256(full_link.encode()).hexdigest()

        author_el = item.select_one("[class*=author__name]")
        author = author_el.get_text(strip=True) if author_el else None

        time_el = item.find("time")
        published = time_el.get("datetime") if time_el else None

        summary_el = item.select_one(".block-text")
        summary = summary_el.get_text(strip=True)[:500] if summary_el else None

        articles.append(
            {
                "source_id": source_id,
                "title": title,
                "author": author,
                "summary": summary,
                "link": full_link,
                "content_hash": content_hash,
                "keywords": None,
                "published_at": published,
            }
        )

    return articles
