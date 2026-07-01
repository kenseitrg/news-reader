"""
LLM-based article summarization and keyword extraction via llama-cpp-python.
"""

import logging
import re
from typing import Any

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.llm = Llama(model_path=model_path, **kwargs)
        self.prompt_template = kwargs.get(
            "prompt",
            "Summarize this news article in 2-3 sentences and extract 3-5 key topics/keywords.\n\nArticle:\n{text}\n\nSummary:\nKeywords:",
        )
        self.max_tokens = kwargs.get("max_tokens", 256)
        self.temperature = kwargs.get("temperature", 0.3)

    def summarize(self, text: str) -> tuple[str, list[str]]:
        prompt = self.prompt_template.format(text=text)
        output = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n"],
        )
        raw = output["choices"][0]["text"].strip()
        return self._parse_output(raw)

    def _parse_output(self, raw: str) -> tuple[str, list[str]]:
        parts = re.split(r"\bKeywords:\s*", raw, maxsplit=1, flags=re.IGNORECASE)
        summary = parts[0].strip() if parts else raw
        keywords = []
        if len(parts) > 1:
            raw_keywords = parts[1]
            keywords = [k.strip().lower() for k in re.split(r"[,;]", raw_keywords) if k.strip()]
        return summary, keywords
