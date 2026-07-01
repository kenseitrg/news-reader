from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return _defaults()
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save(cfg: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _defaults() -> dict[str, Any]:
    return {
        "db_path": str(DATA_DIR / "news_reader.db"),
        "model": {
            "path": "utrobinmv/t5_summary_en_ru_zh_base_2048",
            "n_ctx": 2048,
            "n_threads": 4,
        },
        "summarizer": {
            "max_tokens": 80,
            "temperature": 0.3,
        },
        "ranking": {
            "freshness_weight": 0.4,
            "source_weight": 0.3,
            "keyword_weight": 0.2,
            "author_weight": 0.1,
            "auto_filter_threshold": 0.0,
        },
        "sources": [],
    }
