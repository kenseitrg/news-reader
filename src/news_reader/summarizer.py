"""LLM-based article summarization and keyword extraction via llama-cpp-python."""

import logging
import re
from typing import Any

from llama_cpp import Llama

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant that summarizes news articles.<|im_end|>\n"
    "<|im_start|>user\n"
    "Article: {text}\n\n"
    "Give me a 2-3 sentence summary of this article "
    "and list 3-5 key topics or keywords.\n"
    "Summary:\n"
    "Keywords:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Summarizer:
    """Generate summaries and keywords from article text using a local LLM."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        kwargs: dict[str, Any] = {
            "n_ctx": n_ctx,
            "verbose": False,
        }
        if n_threads:
            kwargs["n_threads"] = n_threads

        self.llm = Llama(model_path=model_path, **kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def summarize(self, text: str) -> tuple[str, list[str]]:
        """Return (summary, list_of_keywords) for the given article text."""
        if not text or not text.strip():
            return "", []

        truncated = self._truncate(text, self.llm.n_ctx() - self.max_tokens - 100)
        prompt = _PROMPT_TEMPLATE.format(text=truncated)

        try:
            output = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<|im_end|>", "\n\n\n"],
            )
            raw = output["choices"][0]["text"].strip()
            return self._parse_output(raw)
        except Exception:
            logger.exception("LLM summarization failed")
            return "", []

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(max_chars - 50, 0)] + "..."

    @staticmethod
    def _parse_output(raw: str) -> tuple[str, list[str]]:
        parts = re.split(r"\bKeywords:\s*", raw, maxsplit=1, flags=re.IGNORECASE)
        summary = parts[0].strip()
        summary = re.sub(r"^Summary:\s*", "", summary, flags=re.IGNORECASE).strip()
        # Strip leftover prompt artifacts
        summary = re.sub(r"\b(key topics?|keywords?)\s*.*", "", summary, flags=re.IGNORECASE).strip()

        keywords: list[str] = []
        if len(parts) > 1:
            raw_keywords = parts[1]
            keywords = [
                k.strip().lower().strip(".")
                for k in re.split(r"[,;]", raw_keywords)
                if k.strip()
            ]

        return summary, keywords
