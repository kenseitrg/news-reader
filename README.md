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
news-reader source-add --name "Meduza" --url https://meduza.io/rss

# Fetch articles (summarises + embeds them)
news-reader fetch

# List unread articles ranked by relevance
news-reader list

# List unread articles sorted by publish date
news-reader list --fresh

# Mark articles as liked / disliked (trains the ranker)
news-reader like 5
news-reader dislike 3

# Re-summarise articles
news-reader summarize --force

# Backfill embeddings for existing articles
news-reader embed
```

## How ranking works

1. Each article is embedded into a 384-dimensional vector using `intfloat/multilingual-e5-small`.
2. When you **like** an article, unread articles with similar embeddings are pushed up.
3. When you **dislike** an article, unread articles with similar embeddings are pushed down.
4. The final score blends embedding affinity with freshness (30-day decay).

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
