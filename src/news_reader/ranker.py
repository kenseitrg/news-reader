"""Article scoring based on embedding similarity and freshness."""

from datetime import datetime, timezone
from typing import Any

from news_reader.embeddings import Embedder


class Ranker:
    """Score and rank unread articles by similarity to liked/disliked articles."""

    def __init__(self, config: dict[str, Any]) -> None:
        weights = config.get("ranking", {})
        self.freshness_weight = weights.get("freshness_weight", 0.3)
        self.embedding_weight = weights.get("embedding_weight", 0.7)
        self.auto_filter_threshold = weights.get("auto_filter_threshold", 0.0)

    def score_articles(
        self,
        articles: list[dict[str, Any]],
        liked_embeddings: list[list[float]],
        disliked_embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Score and sort *articles* in place, returning the same list."""
        for article in articles:
            freshness = self._freshness_score(article.get("published_at"))
            emb_sim = self._embedding_affinity(
                article.get("embedding"),
                liked_embeddings,
                disliked_embeddings,
            )
            article["_score"] = round(
                self.freshness_weight * freshness + self.embedding_weight * emb_sim,
                4,
            )

        articles.sort(key=lambda a: a["_score"], reverse=True)
        return articles

    def _freshness_score(self, published_at: str | None) -> float:
        if not published_at:
            return 0.5
        try:
            pub = datetime.fromisoformat(published_at)
            now = datetime.now(timezone.utc)
            hours = (now - pub).total_seconds() / 3600
            return max(0.0, 1.0 - hours / 720)  # decays over 30 days
        except (ValueError, TypeError):
            return 0.5

    def _embedding_affinity(
        self,
        article_embedding: list[float] | None,
        liked_embeddings: list[list[float]],
        disliked_embeddings: list[list[float]],
    ) -> float:
        if not article_embedding:
            return 0.3

        max_liked = 0.0
        for le in liked_embeddings:
            sim = Embedder.cosine_similarity(article_embedding, le)
            if sim > max_liked:
                max_liked = sim

        max_disliked = 0.0
        for de in disliked_embeddings:
            sim = Embedder.cosine_similarity(article_embedding, de)
            if sim > max_disliked:
                max_disliked = sim

        # liked weight: max_liked (higher is better)
        # disliked penalty: max_disliked (higher is worse)
        raw = max_liked - max_disliked
        # Normalize from [-1, 1] to [0, 1]
        return (raw + 1.0) / 2.0
