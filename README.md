# News Reader

Local CLI news aggregator with multilingual summarisation and personalised ranking.

Fetches articles from RSS feeds and scraped websites, summarises them with a local T5 model, and ranks unread articles by semantic similarity to your likes and dislikes using Sentence-Transformers embeddings.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package & virtualenv manager)

## Setup

```bash
uv sync
uv run news-reader --help
```

## Configuration

Edit `config.yaml`:

```yaml
db_path: data/news_reader.db

model:
  path: utrobinmv/t5_summary_en_ru_zh_base_2048

embedding:
  model: intfloat/multilingual-e5-small

ranking:
  freshness_weight: 0.3
  embedding_weight: 0.7

sources: []
```

## Usage

```bash
# Add an RSS source
news-reader source-add --name "Test" --url https://test.io/rss

# Fetch articles (summarises + embeds them)
news-reader fetch

# List unread articles ranked by relevance
news-reader list

# List unread articles sorted by publish date
news-reader list --fresh

# Mark articles as liked / disliked / read
news-reader like 5       # trains the ranker (positive signal)
news-reader dislike 3    # trains the ranker (negative signal)
news-reader read 8       # dismiss without affecting ranking

# Re-summarise articles
news-reader summarize --force

# Backfill embeddings for existing articles
news-reader embed
```

## How ranking works

Each article is embedded into a 384-dimensional vector using `intfloat/multilingual-e5-small`.

When you mark an article, it gets a score:

| Score | Meaning | Command |
|-------|---------|---------|
| `1` | Liked — similar articles are pushed up | `news-reader like <id>` |
| `-1` | Disliked — similar articles are pushed down | `news-reader dislike <id>` |
| `0` | Read — no effect on ranking | `news-reader read <id>` |

Unread articles are scored by their embedding similarity to liked articles (minus similarity to disliked ones), blended with freshness (30-day decay). Articles marked as read (`0`) are excluded from the list but don't influence the ranker.

This approach works for any language — including Russian — without relying on brittle keyword extraction.

## Project

```
src/news_reader/
├── cli.py          # CLI commands (fetch, list, like, dislike, embed, …)
├── config.py       # YAML config loader
├── embeddings.py   # Sentence-Transformers embedder
├── fetcher.py      # Async RSS / scrape fetchers
├── main.py         # Entry point
├── parser.py       # RSS / HTML parsers
├── ranker.py       # Embedding-based scoring
├── sources.py      # Source dataclass
├── storage.py      # SQLite CRUD
└── summarizer.py   # T5 summarisation model
```

## Development

```bash
uv run ruff check src/
uv run ruff format src/
uv run ty check src/
```
