# Scoring Algorithm Improvement Plan

## Current System Overview

### Database state (as of 2026-07-10)

| Metric | Value |
|---|---|
| Total articles | 582 |
| Liked (+1) | 63 (11%) |
| Disliked (−1) | 254 (44%) |
| Read/no preference (0) | 265 (45%) |
| Articles with embeddings | 297 (51%) |
| Articles with keywords | 159 (27%) |
| Articles with author | 543 (93%) |

### Current formula (`src/news_reader/ranker.py`)

```
score = 0.3 × freshness + 0.7 × embedding_affinity
```

Where:

- **Freshness**: linear decay over 30 days to zero: `max(0, 1 − hours / 720)`
- **Embedding affinity**: `(max_liked_sim − max_disliked_sim + 1) / 2`
  - nearest-neighbour between article embedding and liked/disliked articles
  - symmetric and prone to outliers
  - falls back to 0.5 if no embeddings or no interaction history

### Score tracking tables (unused)

The schema has `source_scores`, `author_scores`, `keyword_scores` tables but none of them is ever written to or read. They are dead schema.

### Missing: `get_new_articles()`

`get_uninteracted_articles()` does a LEFT JOIN on `interactions` — if every article has a row (even score=0), it returns nothing. The ranker never runs on genuinely unseen content.

---

## Signal Semantics

| Interaction | Stored as | Use in ranking |
|---|---|---|
| 👍 Liked | `score = 1` in `interactions` | Positive examples for embedding affinity |
| 🤷 Read / no preference | `score = 0` in `interactions` | Aggregate source & author scoring only (excluded from embedding comparisons) |
| 👎 Disliked | `score = −1` in `interactions` | Negative examples for embedding affinity |
| *(no row)* | `interactions` has no row for this article | Belongs to the **new/unseen** pool — these are the articles scored and surfaced users |

score=0 rows are kept — they provide useful training signal for source and author preferences. They are simply excluded from the "new" query so they don't appear in the ranked list again.

---

## Plan

### Phase A — Fix the data pipeline so the ranker can run

**A1. Add `get_new_articles()`** storage method

Mirrors existing `get_uninteracted_articles()` — articles with no interaction row at all. This returns the (~0 now, will grow with future fetches) pool of unseen content.

**A2. Update list command to use `get_new_articles()`**

Both `cli.py` and `web.py` switch to the new method.

**A3. Populate missing embeddings**

285 articles lack embeddings. Run `news-reader embed` to fill them.

---

### Phase B — Write score aggregations on interaction

**B1. Expand `set_interaction()`** to update score tracking tables

When a user likes or dislikes an article, upsert into:

- `source_scores` by `source_id` — increment likes/dislikes, recompute score
- `author_scores` by `(author, source_id)` — same

Formula for the aggregate score column:

```sql
score = ROUND(1.0 * (likes − dislikes) / (likes + dislikes), 4)  -- range [−1, 1]
```

score=0 rows do NOT trigger these updates (no opinion expressed).

---

### Phase C — Build multi-component scoring (ranker.py)

New formula:

```
score = w_f × freshness + w_s × source + w_a × author + w_e × embedding
```

Weights default to:

```python
freshness = 0.25
source    = 0.25
author    = 0.25
embedding = 0.25
```

**C1. Source affinity**

Look up `source_scores` by `source_id`. Normalize to [0, 1]:

```python
raw = (likes − dislikes) / (likes + dislikes)     # [−1, 1]
affinity = (raw + 1.0) / 2.0                       # [0, 1]
```

Fallback to 0.5 if no data.

**C2. Author affinity**

Look up `author_scores` by `(author, source_id)`. Same normalization.

Fallback to 0.5 if author is unknown or absent.

**C3. Improved embedding affinity**

Replace nearest-neighbour with equally-weighted average:

```python
liked_mean    = mean(sim(article, each liked))     if any liked  else 0.0
disliked_mean = mean(sim(article, each disliked))  if any dislik else 0.0
raw           = liked_mean − disliked_mean                       # [−1, 1]
affinity      = (raw + 1.0) / 2.0                                 # [0, 1]
```

Fallback to 0.5 if no embeddings or no history.

**C4. Freshness (unchanged)**

Keep current linear-decay model for now.

**C5. Apply `auto_filter_threshold`**

After computing scores, filter out articles where `_score < threshold` (default 0.0).

---

### Phase D — Update configuration

**D1. Update `config.yaml`**

```yaml
ranking:
  freshness_weight: 0.25
  source_weight: 0.25
  author_weight: 0.25
  embedding_weight: 0.25
  auto_filter_threshold: 0.0
```

**D2. Update `config.py` defaults** to match.

---

### Phase E — Polish

**E1. `news-reader stats` command**

Show personalization stats:
- Top sources by score
- Top authors by score
- Total likes/dislikes distribution

**E2. Add a static analysis helper script**

`scripts/score-stats.sh` — quick SQLite queries to inspect score distributions.

---

## Files affected

| File | Change |
|---|---|
| `src/news_reader/storage.py` | Add `get_new_articles()`, expand `set_interaction()`, add helper stats queries |
| `src/news_reader/ranker.py` | Refactor: new multi-component score, better embedding math, threshold filtering |
| `src/news_reader/cli.py` | Use `get_new_articles()`, add `stats` command |
| `src/news_reader/web.py` | Use `get_new_articles()` |
| `src/news_reader/config.py` | Update default weights (0.25 each) |
| `config.yaml` | Update weights |
| `docs/scoring-improvement.md` | This file |
| `scripts/score-stats.sh` | SQLite diagnostic script |

---

## Migration Note

No schema changes required. Existing data is preserved. score=0 rows stay in `interactions` and are used for source/author scoring but NOT for embedding affinity or for listing.

---

## Future considerations (not in scope)

- Exponential freshness decay (more differentiation among recent articles)
- Bayesian smoothing for source/author scores with few samples
- Per-source embedding sub-clusters for diverse articles from same source
- Re-scoring pass for all articles when config weights change
