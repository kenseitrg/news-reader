"""
Article content parsing utilities.
Uses feedparser for RSS/Atom feeds and trafilatura for HTML page extraction.
"""

import hashlib

import feedparser
from trafilatura import extract


def parse_rss(xml_content: str, source_id: int) -> list[dict]:
    feed = feedparser.parse(xml_content)
    articles = []
    for entry in feed.entries:
        link = entry.get("link", "")
        content_hash = hashlib.sha256(link.encode()).hexdigest()
        tags = []
        if hasattr(entry, "tags"):
            tags = [t.get("term", "") for t in entry.tags if t.get("term")]

        articles.append({
            "source_id": source_id,
            "title": entry.get("title", ""),
            "author": entry.get("author"),
            "summary": entry.get("summary"),
            "link": link,
            "content_hash": content_hash,
            "keywords": ",".join(tags) if tags else None,
            "published_at": entry.get("published"),
        })
    return articles


def parse_html(html_content: str) -> str | None:
    return extract(html_content)
