"""Multilingual article embeddings via Sentence-Transformers."""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

E5_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    """Generate embeddings for article text using a Sentence-Transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or E5_MODEL
        self._model: Any = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model %s ...", self._model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded")

    def embed(self, text: str) -> list[float]:
        """Produce a 384‑dim embedding vector for *text*.

        The ``"passage: "`` prefix required by E5 is added automatically.
        """
        self._lazy_load()
        prefixed = f"passage: {text}"
        vec = self._model.encode(prefixed, normalize_embeddings=True)
        return vec.tolist()

    def embed_article(self, title: str, summary: str) -> list[float]:
        """Embed an article from its *title* and *summary* text."""
        text = f"{title}\n\n{summary}" if summary else title
        return self.embed(text)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
