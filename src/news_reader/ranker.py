"""Article scoring based on embedding similarity and freshness."""

from datetime import datetime, timezone
from typing import Any


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
        import numpy as np

        has_liked = liked_embeddings and liked_embeddings[0]
        has_disliked = disliked_embeddings and disliked_embeddings[0]

        liked_arr = (
            np.asarray(liked_embeddings, dtype=np.float32) if has_liked else None
        )
        disliked_arr = (
            np.asarray(disliked_embeddings, dtype=np.float32) if has_disliked else None
        )

        for article in articles:
            freshness = self._freshness_score(article.get("published_at"))
            emb_sim = self._embedding_affinity(
                article.get("embedding"),
                liked_arr,
                disliked_arr,
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
        liked_arr: Any | None,
        disliked_arr: Any | None,
    ) -> float:
        if article_embedding is None or (liked_arr is None and disliked_arr is None):
            return 0.5

        import numpy as np

        vec = np.asarray(article_embedding, dtype=np.float32)

        max_liked = 0.0
        if liked_arr is not None and liked_arr.shape[0] > 0:
            max_liked = float((liked_arr @ vec).max())

        max_disliked = 0.0
        if disliked_arr is not None and disliked_arr.shape[0] > 0:
            max_disliked = float((disliked_arr @ vec).max())

        raw = max_liked - max_disliked
        return (raw + 1.0) / 2.0
