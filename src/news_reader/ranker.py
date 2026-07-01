"""
Article scoring and personalization based on user interactions.
"""

from datetime import datetime, timezone
from typing import Any


class Ranker:
    def __init__(self, config: dict[str, Any]) -> None:
        weights = config.get("ranking", {})
        self.freshness_weight = weights.get("freshness_weight", 0.4)
        self.source_weight = weights.get("source_weight", 0.3)
        self.keyword_weight = weights.get("keyword_weight", 0.2)
        self.author_weight = weights.get("author_weight", 0.1)
        self.auto_filter_threshold = weights.get("auto_filter_threshold", 0.0)

    def score_article(
        self,
        article: dict[str, Any],
        source_score: float,
        keyword_scores: dict[str, float],
        author_score: float,
    ) -> float:
        freshness = self._freshness_score(article.get("published_at"))
        source = self._normalize(source_score)
        keyword = self._keyword_affinity(article.get("keywords", ""), keyword_scores)
        author = self._normalize(author_score)

        total = (
            self.freshness_weight * freshness
            + self.source_weight * source
            + self.keyword_weight * keyword
            + self.author_weight * author
        )
        return round(total, 4)

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

    def _normalize(self, score: float) -> float:
        # normalize a score from [-1, 1] range to [0, 1]
        return (score + 1) / 2

    def _keyword_affinity(self, keywords_str: str, keyword_scores: dict[str, float]) -> float:
        if not keywords_str:
            return 0.5
        words = [w.strip().lower() for w in keywords_str.replace(",", " ").split() if w.strip()]
        if not words:
            return 0.5
        scores = [keyword_scores.get(w, 0.0) for w in words]
        avg = sum(scores) / len(scores)
        return self._normalize(avg)
