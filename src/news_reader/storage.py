import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Storage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL CHECK(type IN ('rss', 'scrape')),
                    url TEXT NOT NULL,
                    feed_url TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER REFERENCES sources(id),
                    title TEXT NOT NULL,
                    author TEXT,
                    summary TEXT,
                    link TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    keywords TEXT,
                    published_at TEXT,
                    fetched_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                    liked INTEGER CHECK(liked IN (0, 1)),
                    clicked_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS source_scores (
                    source_id INTEGER PRIMARY KEY REFERENCES sources(id),
                    score REAL DEFAULT 0.0,
                    likes INTEGER DEFAULT 0,
                    dislikes INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS author_scores (
                    author TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    score REAL DEFAULT 0.0,
                    likes INTEGER DEFAULT 0,
                    dislikes INTEGER DEFAULT 0,
                    PRIMARY KEY (author, source_id)
                );

                CREATE TABLE IF NOT EXISTS keyword_scores (
                    keyword TEXT NOT NULL,
                    score REAL DEFAULT 0.0,
                    likes INTEGER DEFAULT 0,
                    dislikes INTEGER DEFAULT 0,
                    PRIMARY KEY (keyword)
                );
            """)

    # --- sources ---

    def add_source(self, name: str, type_: str, url: str, feed_url: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sources (name, type, url, feed_url) VALUES (?, ?, ?, ?)",
                (name, type_, url, feed_url),
            )
            return cur.lastrowid

    def get_sources(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        with self._conn() as conn:
            query = "SELECT * FROM sources"
            if enabled_only:
                query += " WHERE enabled = 1"
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def remove_source(self, source_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    # --- articles ---

    def article_exists(self, link: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM articles WHERE link = ?", (link,)
            ).fetchone()
            return row is not None

    def add_article(self, article: dict[str, Any]) -> int | None:
        if self.article_exists(article["link"]):
            return None
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO articles
                   (source_id, title, author, summary, link, content_hash, keywords, published_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article["source_id"],
                    article["title"],
                    article.get("author"),
                    article.get("summary"),
                    article["link"],
                    article["content_hash"],
                    article.get("keywords"),
                    article.get("published_at"),
                ),
            )
            return cur.lastrowid

    def get_uninteracted_articles(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT a.*, s.name as source_name
                   FROM articles a
                   JOIN sources s ON a.source_id = s.id
                   LEFT JOIN interactions i ON a.id = i.article_id
                   WHERE i.article_id IS NULL
                   ORDER BY a.published_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    # --- interactions ---

    def set_interaction(self, article_id: int, liked: bool | None) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO interactions (article_id, liked, clicked_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(article_id) DO UPDATE SET liked = ?, clicked_at = datetime('now')""",
                (article_id, liked, liked),
            )
