"""Article summarization and keyword extraction using T5 models."""

import logging
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

SUM_MODEL = "utrobinmv/t5_summary_en_ru_zh_base_2048"
KW_MODEL = "agentlans/flan-t5-small-keywords"


class Summarizer:
    """Generate summaries and extract keywords from article text."""

    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_tokens: int = 80,
        temperature: float = 0.3,
    ) -> None:
        name = model_path or SUM_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Summarization model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device)

        # Keyword extraction model (lazy-loaded on first use)
        self._kw_tokenizer: Any = None
        self._kw_model: Any = None

        self.max_input_length = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.model.generation_config.max_length = None
        self.model.generation_config.max_new_tokens = None

    def _lazy_load_kw(self) -> None:
        if self._kw_model is not None:
            return
        self._kw_tokenizer = AutoTokenizer.from_pretrained(KW_MODEL)
        self._kw_model = AutoModelForSeq2SeqLM.from_pretrained(KW_MODEL).to(self.device)

    def summarize(self, text: str) -> tuple[str, list[str]]:
        """Return (summary, list_of_keywords) for the given article text."""
        if not text or not text.strip():
            return "", []

        summary = self._generate_summary(text)
        keywords = self._extract_keywords(text)
        return summary, keywords

    def _generate_summary(self, text: str) -> str:
        prompt = f"summary: {text}"
        return self._generate(self.tokenizer, self.model, prompt, self.max_tokens)

    def _extract_keywords(self, text: str) -> list[str]:
        self._lazy_load_kw()
        # FLAN-T5-small max position embeddings = 512
        prompt = text
        raw = self._generate(self._kw_tokenizer, self._kw_model, prompt, 128, 512)
        if not raw:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for kw in raw.split("||"):
            kw = kw.strip().lower()
            if kw and kw not in seen:
                seen.add(kw)
                result.append(kw)
        return result[:10]

    def _generate(
        self,
        tokenizer: Any,
        model: Any,
        prompt: str,
        max_new: int,
        max_input_length: int | None = None,
    ) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_input_length or self.max_input_length,
            truncation=True,
        ).to(self.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "num_beams": 2,
            "early_stopping": True,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
