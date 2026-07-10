"""Article scoring based on freshness, source/author affinity, and embedding similarity."""

from datetime import datetime, timezone
from typing import Any

import numpy as np


def _normalize(raw: float) -> float:
    """Map [-1, 1] -> [0, 1]."""
    return (raw + 1.0) / 2.0


class Ranker:
    """Score and rank unread articles by multiple personalisation signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        weights = config.get("ranking", {})
        self.freshness_weight = weights.get("freshness_weight", 0.25)
        self.source_weight = weights.get("source_weight", 0.25)
        self.author_weight = weights.get("author_weight", 0.25)
        self.embedding_weight = weights.get("embedding_weight", 0.25)
        self.auto_filter_threshold = weights.get("auto_filter_threshold", 0.0)

    def score_articles(
        self,
        articles: list[dict[str, Any]],
        liked_embeddings: list[list[float]],
        disliked_embeddings: list[list[float]],
        source_scores: dict[int, dict[str, Any]] | None = None,
        author_scores: dict[tuple[str, int], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Score and sort *articles* in place, returning the same list.

        Each article receives a ``_score`` in [0, 1] computed as a weighted
        sum of four components: freshness, source affinity, author affinity,
        and embedding similarity.
        """
        has_liked = bool(liked_embeddings and liked_embeddings[0])
        has_disliked = bool(disliked_embeddings and disliked_embeddings[0])

        liked_arr = (
            np.asarray(liked_embeddings, dtype=np.float32) if has_liked else None
        )
        disliked_arr = (
            np.asarray(disliked_embeddings, dtype=np.float32) if has_disliked else None
        )

        for article in articles:
            freshness = self._freshness_score(article.get("published_at"))
            source_aff = self._source_affinity(article.get("source_id"), source_scores)
            author_aff = self._author_affinity(
                article.get("author"),
                article.get("source_id"),
                author_scores,
            )
            emb_sim = self._embedding_affinity(
                article.get("embedding"), liked_arr, disliked_arr
            )

            article["_score"] = round(
                self.freshness_weight * freshness
                + self.source_weight * source_aff
                + self.author_weight * author_aff
                + self.embedding_weight * emb_sim,
                4,
            )

        articles.sort(key=lambda a: a["_score"], reverse=True)

        if self.auto_filter_threshold > 0.0:
            articles[:] = [
                a for a in articles if a["_score"] >= self.auto_filter_threshold
            ]

        return articles

    # ------------------------------------------------------------------
    # Component scorers – each returns a float in [0, 1]
    # ------------------------------------------------------------------

    @staticmethod
    def _freshness_score(published_at: str | None) -> float:
        """Linear decay from 1.0 (just published) to 0.0 (30+ days old)."""
        if not published_at:
            return 0.5
        try:
            pub = datetime.fromisoformat(published_at)
            now = datetime.now(timezone.utc)
            hours = (now - pub).total_seconds() / 3600
            return max(0.0, 1.0 - hours / 720)  # 720 h = 30 days
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _source_affinity(
        source_id: int | None,
        source_scores: dict[int, dict[str, Any]] | None,
    ) -> float:
        if source_id is None or not source_scores:
            return 0.5
        row = source_scores.get(source_id)
        if row is None or row.get("likes", 0) + row.get("dislikes", 0) == 0:
            return 0.5
        raw = row["score"]  # already in [-1, 1]
        return _normalize(raw)

    @staticmethod
    def _author_affinity(
        author: str | None,
        source_id: int | None,
        author_scores: dict[tuple[str, int], dict[str, Any]] | None,
    ) -> float:
        if not author or source_id is None or not author_scores:
            return 0.5
        row = author_scores.get((author, source_id))
        if row is None or row.get("likes", 0) + row.get("dislikes", 0) == 0:
            return 0.5
        raw = row["score"]
        return _normalize(raw)

    @staticmethod
    def _embedding_affinity(
        article_embedding: list[float] | None,
        liked_arr: Any | None,
        disliked_arr: Any | None,
    ) -> float:
        """Mean cosine-similarity to liked minus mean similarity to disliked."""
        if article_embedding is None or (liked_arr is None and disliked_arr is None):
            return 0.5

        vec = np.asarray(article_embedding, dtype=np.float32)

        liked_mean = 0.0
        if liked_arr is not None and liked_arr.shape[0] > 0:
            liked_mean = float((liked_arr @ vec).mean())

        disliked_mean = 0.0
        if disliked_arr is not None and disliked_arr.shape[0] > 0:
            disliked_mean = float((disliked_arr @ vec).mean())

        raw = liked_mean - disliked_mean  # [-1, 1]
        return _normalize(raw)
