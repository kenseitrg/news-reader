# News Reader — Implementation Plan

## Overview

A local Python application that aggregates news articles from multiple sources (RSS feeds + web scraping), generates short summaries using a local LLM, and learns user preferences through like/dislike feedback. CLI-first, with future Telegram bot support.

---

## Project Structure

```
news-reader/
├── pyproject.toml              # project metadata & dependencies
├── config.yaml                 # sources, model settings, etc.
├── PLAN.md                     # this file
├── README.md
├── data/
│   └── news_reader.db          # SQLite database (auto-created)
└── src/
    └── news_reader/
        ├── __init__.py
        ├── main.py             # CLI entry point
        ├── config.py           # load/save config.yaml
        ├── sources.py          # source management (add/remove/list)
        ├── fetcher.py          # async HTTP fetching (httpx)
        ├── parser.py           # RSS (feedparser) + HTML (trafilatura) parsing
        ├── summarizer.py       # LLM summarization via llama-cpp-python
        ├── storage.py          # SQLite CRUD operations
        ├── ranker.py           # article scoring & personalization
        └── cli.py              # typer commands + rich output
```

---

## Components

### 1. config.py
- Load/save `config.yaml` using `pyyaml`
- Stores: list of sources, LLM model path/params, default settings

### 2. sources.py
- Source model: `{name, type: rss|scrape, url, feed_url, enabled}`
- `RSSSource` — standard RSS/Atom feed URL
- `ScrapeSource` — a page URL where articles are listed (uses BS4 + selectors if needed)
- Add/remove/enable/disable sources

### 3. fetcher.py
- Async HTTP via `httpx`
- Fetches RSS XML or HTML page for each enabled source
- Returns raw content bytes for the parser

### 4. parser.py
- **RSS**: `feedparser` → extract title, author, summary, link, published date, tags/categories
- **HTML (scrape)**: `trafilatura` for article content extraction from full pages; `BeautifulSoup4` for list-of-articles scraping (if needed)
- Extracts metadata: author, keywords/tags, publication date
- Deduplication by link content hash

### 5. summarizer.py
- Uses `transformers` with two models:
  - **utrobinmv/t5_summary_en_ru_zh_base_2048** (T5-base, 0.3B params) — summarization in EN/RU/ZH via `summary:` prefix
  - **agentlans/flan-t5-small-keywords** (FLAN-T5-small, 77M params) — English keyword extraction, lazy-loaded on first use
- Both models cached locally after first download (~1GB total on disk)
- Fast on CPU (~1-5s per article for summary, +1-2s for keywords)

### 6. storage.py
- **SQLite** via `sqlite3` (stdlib)
- Tables:

```sql
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('rss', 'scrape')),
    url TEXT NOT NULL,
    feed_url TEXT,
    enabled INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER REFERENCES sources(id),
    title TEXT NOT NULL,
    author TEXT,
    summary TEXT,
    link TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    keywords TEXT,              -- comma-separated or JSON array
    published_at TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE interactions (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    liked INTEGER CHECK(liked IN (0, 1)),
    clicked_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE source_scores (
    source_id INTEGER PRIMARY KEY REFERENCES sources(id),
    score REAL DEFAULT 0.0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0
);

CREATE TABLE author_scores (
    author TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    score REAL DEFAULT 0.0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    PRIMARY KEY (author, source_id)
);

CREATE TABLE keyword_scores (
    keyword TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    likes INTEGER DEFAULT 0,
    dislikes INTEGER DEFAULT 0,
    PRIMARY KEY (keyword)
);
```

### 7. ranker.py
- Scores each article based on:
  - **Freshness** — exponential decay from publish date
  - **Source score** — average liked/disliked ratio for that source
  - **Author affinity** — boost/drop articles from authors matching liked/disliked history
  - **Keyword/tag affinity** — TF-IDF-like matching of keywords and title words against liked articles; penalizes tags consistently disliked
  - **Category affinity** — same for RSS categories/tags
- Returns sorted list of articles for display
- **Auto-filtering**: once enough interactions are accumulated, articles falling below a configurable score threshold are automatically dropped/hidden from the list
- Updates source, author, and keyword scores on new interactions

### 8. cli.py
- **`news-reader fetch`** — fetch + parse + summarize new articles
- **`news-reader list`** — show unscored/unread articles with rank
- **`news-reader like <id>`** / **`news-reader dislike <id>`** — give feedback
- **`news-reader sources`** — list/disable/enable sources
- **`news-reader source add --name X --url Y --type rss`** — add source
- **`news-reader source remove <id>`** — remove source

Output via `rich` Table:
```
 # │ Source       │ Title                          │ Rank
───┼──────────────┼────────────────────────────────┼───────
 1 │ BBC News     │ UK climate targets announced   │ ★★★★☆
 2 │ TechCrunch   │ GPT-5 released                 │ ★★★☆☆
 3 │ Reuters      │ Oil prices drop                │ ★★☆☆☆
```

Selecting an article shows full summary + link.

---

## Dependencies (pyproject.toml)

```toml
dependencies = [
    "httpx>=0.27",
    "feedparser>=6.0",
    "trafilatura>=1.6",
    "beautifulsoup4>=4.12",
    "transformers>=4.30",
    "torch>=2.0",
    "protobuf>=3.20",
    "sentencepiece>=0.1",
    "protobuf>=3.20",
    "sentencepiece>=0.1",
    "typer>=0.9",
    "rich>=13.0",
    "pyyaml>=6.0",
]
```

---

## Implementation Phases

### Phase 1 — Core MVP
1. Project scaffolding (`pyproject.toml`, package structure)
2. `config.py` — YAML load/save
3. `storage.py` — SQLite init + CRUD
4. `sources.py` — source management
5. `fetcher.py` + `parser.py` — RSS fetching and parsing
6. `summarizer.py` — llama-cpp-python integration
7. `ranker.py` — basic freshness + source score
8. `cli.py` — all commands working
9. `main.py` — wire everything together

### Phase 2 — Personalization
1. Author, keyword/tag, and category affinity in ranker
2. Interaction history analytics (per-author, per-keyword, per-source stats)
3. Auto-filtering: hide articles below configurable score threshold
4. Configurable ranking weights (freshness vs affinity)
5. CLI command to show personalization stats (top authors, top keywords, etc.)

### Phase 3 — Telegram Bot
1. `bot.py` module using `python-telegram-bot`
2. Reuse core modules (fetcher, storage, ranker)
3. Bot commands mirroring CLI commands
4. Inline keyboards for like/dislike

---

## Future Considerations

- **Scrape sources**: per-site CSS selectors stored in config for parsing article lists
- **Caching**: HTTP caching for RSS feeds to avoid re-downloading unchanged content
- **Scheduling**: periodic `fetch` via cron/systemd timer (or built-in scheduler)
- **Search**: full-text search over articles
- **Read-it-later / mark-as-read**: track read status
- **Export/Import**: sources and interactions as JSON
