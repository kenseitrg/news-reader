import sqlite3
import struct
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(data: bytes) -> list[float]:
    return list(struct.unpack(f"{len(data) // 4}f", data))


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
            # Check if schema already exists
            has_articles = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
            ).fetchone()

            if has_articles is None:
                conn.executescript("""
                    CREATE TABLE sources (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL CHECK(type IN ('rss', 'scrape')),
                        url TEXT NOT NULL,
                        feed_url TEXT,
                        enabled INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT (datetime('now'))
                    );

                    CREATE TABLE articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id INTEGER REFERENCES sources(id),
                        title TEXT NOT NULL,
                        author TEXT,
                        summary TEXT,
                        link TEXT NOT NULL UNIQUE,
                        content_hash TEXT NOT NULL,
                        keywords TEXT,
                        embedding BLOB,
                        published_at TEXT,
                        fetched_at TEXT DEFAULT (datetime('now'))
                    );

                    CREATE TABLE interactions (
                        article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                        score INTEGER CHECK(score IN (-1, 0, 1)),
                        clicked_at TEXT,
                        created_at TEXT DEFAULT (datetime('now'))
                    );

                    CREATE TABLE source_scores (
                        source_id INTEGER PRIMARY KEY REFERENCES sources(id),
                        score REAL DEFAULT 0.0,
                        likes INTEGER DEFAULT 0,
                        dislikes INTEGER DEFAULT 0
                    );

                    CREATE TABLE author_scores (
                        author TEXT NOT NULL,
                        source_id INTEGER NOT NULL,
                        score REAL DEFAULT 0.0,
                        likes INTEGER DEFAULT 0,
                        dislikes INTEGER DEFAULT 0,
                        PRIMARY KEY (author, source_id)
                    );

                    CREATE TABLE keyword_scores (
                        keyword TEXT NOT NULL,
                        score REAL DEFAULT 0.0,
                        likes INTEGER DEFAULT 0,
                        dislikes INTEGER DEFAULT 0,
                        PRIMARY KEY (keyword)
                    );
                """)
                return

            # Migration: add embedding column for pre-existing databases
            cols = [col[1] for col in conn.execute("PRAGMA table_info(articles)")]
            if "embedding" not in cols:
                conn.execute("ALTER TABLE articles ADD COLUMN embedding BLOB")

            # Migration: liked → score for pre-existing databases
            icols = {col[1] for col in conn.execute("PRAGMA table_info(interactions)")}
            if "liked" in icols:
                conn.executescript("""
                    ALTER TABLE interactions RENAME TO interactions_old;
                    CREATE TABLE interactions (
                        article_id INTEGER PRIMARY KEY REFERENCES articles(id),
                        score INTEGER CHECK(score IN (-1, 0, 1)),
                        clicked_at TEXT,
                        created_at TEXT DEFAULT (datetime('now'))
                    );
                    INSERT INTO interactions (article_id, score, clicked_at, created_at)
                        SELECT article_id,
                               CASE WHEN liked = 0 THEN -1 ELSE liked END,
                               clicked_at, created_at
                        FROM interactions_old;
                    DROP TABLE interactions_old;
                """)

            # Migration: clear HN placeholder summaries so LLM can regenerate them
            hn_fixed = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE summary LIKE 'Score:%'"
            ).fetchone()[0]
            if hn_fixed:
                conn.execute(
                    "UPDATE articles SET summary = NULL WHERE summary LIKE 'Score:%'"
                )

            # Migration: backfill source_scores and author_scores from existing interactions
            has_source_scores = conn.execute(
                "SELECT COUNT(*) FROM source_scores"
            ).fetchone()[0]
            if has_source_scores == 0:
                conn.executescript("""
                    INSERT INTO source_scores (source_id, likes, dislikes, score)
                    SELECT
                        a.source_id,
                        COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)           AS likes,
                        COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0)          AS dislikes,
                        ROUND(
                            1.0 * (
                                COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)
                                -
                                COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0)
                            )
                            / NULLIF(
                                COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)
                                +
                                COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0),
                                0
                            ),
                            4
                        )                                                                   AS score
                    FROM articles a
                    JOIN interactions i ON a.id = i.article_id
                    WHERE i.score IN (1, -1)
                    GROUP BY a.source_id;
                """)

            has_author_scores = conn.execute(
                "SELECT COUNT(*) FROM author_scores"
            ).fetchone()[0]
            if has_author_scores == 0:
                conn.executescript("""
                    INSERT INTO author_scores (author, source_id, likes, dislikes, score)
                    SELECT
                        a.author,
                        a.source_id,
                        COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)           AS likes,
                        COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0)          AS dislikes,
                        ROUND(
                            1.0 * (
                                COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)
                                -
                                COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0)
                            )
                            / NULLIF(
                                COALESCE(SUM(CASE WHEN i.score = 1 THEN 1 ELSE 0 END), 0)
                                +
                                COALESCE(SUM(CASE WHEN i.score = -1 THEN 1 ELSE 0 END), 0),
                                0
                            ),
                            4
                        )                                                                   AS score
                    FROM articles a
                    JOIN interactions i ON a.id = i.article_id
                    WHERE i.score IN (1, -1)
                      AND a.author IS NOT NULL AND a.author != ''
                    GROUP BY a.author, a.source_id;
                """)

    # --- sources ---

    def add_source(
        self, name: str, type_: str, url: str, feed_url: str | None = None
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sources (name, type, url, feed_url) VALUES (?, ?, ?, ?)",
                (name, type_, url, feed_url),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def get_sources(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        with self._conn() as conn:
            query = "SELECT * FROM sources"
            if enabled_only:
                query += " WHERE enabled = 1"
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def set_source_enabled(self, source_id: int, enabled: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sources SET enabled = ? WHERE id = ?",
                (int(enabled), source_id),
            )

    def remove_source(self, source_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM interactions WHERE article_id IN "
                "(SELECT id FROM articles WHERE source_id = ?)",
                (source_id,),
            )
            conn.execute("DELETE FROM articles WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM source_scores WHERE source_id = ?", (source_id,))
            conn.execute(
                "DELETE FROM author_scores WHERE source_id = ?",
                (source_id,),
            )
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

    def get_new_articles(self) -> list[dict[str, Any]]:
        """Articles that have no interaction row — the unseen/unranked pool."""
        return self.get_uninteracted_articles()

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
            articles = [dict(r) for r in rows]
            for a in articles:
                blob = a.get("embedding")
                if blob and isinstance(blob, bytes):
                    a["embedding"] = _deserialize_embedding(blob)
            return articles

    # --- interactions ---

    def get_articles_without_summaries(
        self, force: bool = False
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if force:
                rows = conn.execute(
                    """SELECT a.*, s.name as source_name
                       FROM articles a
                       JOIN sources s ON a.source_id = s.id
                       ORDER BY a.published_at DESC"""
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT a.*, s.name as source_name
                       FROM articles a
                       JOIN sources s ON a.source_id = s.id
                       WHERE a.summary IS NULL OR a.summary = ''
                       ORDER BY a.published_at DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

    def update_article_summary(
        self, article_id: int, summary: str, keywords: str | None
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE articles SET summary = ?, keywords = ? WHERE id = ?",
                (summary, keywords, article_id),
            )

    def set_interaction(self, article_id: int, score: int) -> None:
        with self._conn() as conn:
            # Look up article metadata needed for score tracking
            article = conn.execute(
                "SELECT source_id, author FROM articles WHERE id = ?",
                (article_id,),
            ).fetchone()
            if article is None:
                return

            # Read old interaction score so we can reverse it
            old = conn.execute(
                "SELECT score FROM interactions WHERE article_id = ?",
                (article_id,),
            ).fetchone()
            old_score: int | None = old["score"] if old else None

            # Update interaction row (existing behaviour)
            conn.execute(
                """INSERT INTO interactions (article_id, score, clicked_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(article_id) DO UPDATE SET
                       score = ?, clicked_at = datetime('now')""",
                (article_id, score, score),
            )

            # Only polar scores (1 / -1) affect aggregate tracking
            if score == 0:
                self._reverse_score_contribution(conn, article, old_score)
                return

            self._reverse_score_contribution(conn, article, old_score)
            self._apply_score_contribution(conn, article, score)

    @staticmethod
    def _reverse_score_contribution(
        conn: sqlite3.Connection,
        article: sqlite3.Row,
        old_score: int | None,
    ) -> None:
        """Undo the old_score contribution before applying a new one."""
        source_id = article["source_id"]
        author: str | None = article["author"]

        if old_score == 1:
            conn.execute(
                """UPDATE source_scores SET
                       likes = MAX(likes - 1, 0),
                       score = COALESCE(
                           ROUND(1.0 * (likes - 1 - dislikes) / NULLIF(likes - 1 + dislikes, 0), 4),
                           0.0
                       )
                   WHERE source_id = ?""",
                (source_id,),
            )
            if author:
                conn.execute(
                    """UPDATE author_scores SET
                           likes = MAX(likes - 1, 0),
                           score = COALESCE(
                               ROUND(1.0 * (likes - 1 - dislikes) / NULLIF(likes - 1 + dislikes, 0), 4),
                               0.0
                           )
                       WHERE author = ? AND source_id = ?""",
                    (author, source_id),
                )
        elif old_score == -1:
            conn.execute(
                """UPDATE source_scores SET
                       dislikes = MAX(dislikes - 1, 0),
                       score = COALESCE(
                           ROUND(1.0 * (likes - dislikes + 1) / NULLIF(likes + dislikes - 1, 0), 4),
                           0.0
                       )
                   WHERE source_id = ?""",
                (source_id,),
            )
            if author:
                conn.execute(
                    """UPDATE author_scores SET
                           dislikes = MAX(dislikes - 1, 0),
                           score = COALESCE(
                               ROUND(1.0 * (likes - dislikes + 1) / NULLIF(likes + dislikes - 1, 0), 4),
                               0.0
                           )
                       WHERE author = ? AND source_id = ?""",
                    (author, source_id),
                )

    @staticmethod
    def _apply_score_contribution(
        conn: sqlite3.Connection,
        article: sqlite3.Row,
        score: int,
    ) -> None:
        """Record a new polar interaction (score is 1 or -1)."""
        source_id = article["source_id"]
        author: str | None = article["author"]

        if score == 1:
            conn.execute(
                """INSERT INTO source_scores (source_id, likes, dislikes, score)
                       VALUES (?, 1, 0, 1.0)
                       ON CONFLICT(source_id) DO UPDATE SET
                           likes = likes + 1,
                           score = ROUND(1.0 * (likes + 1 - dislikes) / NULLIF(likes + 1 + dislikes, 0), 4)""",
                (source_id,),
            )
            if author:
                conn.execute(
                    """INSERT INTO author_scores (author, source_id, likes, dislikes, score)
                           VALUES (?, ?, 1, 0, 1.0)
                           ON CONFLICT(author, source_id) DO UPDATE SET
                               likes = likes + 1,
                               score = ROUND(1.0 * (likes + 1 - dislikes) / NULLIF(likes + 1 + dislikes, 0), 4)""",
                    (author, source_id),
                )
        elif score == -1:
            conn.execute(
                """INSERT INTO source_scores (source_id, likes, dislikes, score)
                       VALUES (?, 0, 1, -1.0)
                       ON CONFLICT(source_id) DO UPDATE SET
                           dislikes = dislikes + 1,
                           score = ROUND(1.0 * (likes - dislikes - 1) / NULLIF(likes + dislikes + 1, 0), 4)""",
                (source_id,),
            )
            if author:
                conn.execute(
                    """INSERT INTO author_scores (author, source_id, likes, dislikes, score)
                           VALUES (?, ?, 0, 1, -1.0)
                           ON CONFLICT(author, source_id) DO UPDATE SET
                               dislikes = dislikes + 1,
                               score = ROUND(1.0 * (likes - dislikes - 1) / NULLIF(likes + dislikes + 1, 0), 4)""",
                    (author, source_id),
                )

    # --- embeddings ---

    def update_article_embedding(self, article_id: int, embedding: list[float]) -> None:
        blob = _serialize_embedding(embedding)
        with self._conn() as conn:
            conn.execute(
                "UPDATE articles SET embedding = ? WHERE id = ?",
                (blob, article_id),
            )

    def get_articles_without_embeddings(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT a.*, s.name as source_name
                   FROM articles a
                   JOIN sources s ON a.source_id = s.id
                   WHERE a.embedding IS NULL
                   ORDER BY a.published_at DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    # --- score tracking tables ---

    def get_source_scores(self) -> dict[int, dict[str, Any]]:
        """Return source_id -> {score, likes, dislikes} for sources with data."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id, score, likes, dislikes FROM source_scores"
            ).fetchall()
            return {r["source_id"]: dict(r) for r in rows}

    def get_author_scores(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Return (author, source_id) -> {score, likes, dislikes}."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT author, source_id, score, likes, dislikes FROM author_scores"
            ).fetchall()
            return {(r["author"], r["source_id"]): dict(r) for r in rows}

    def get_interacted_articles(self, score: int) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT a.*, s.name as source_name
                   FROM articles a
                   JOIN sources s ON a.source_id = s.id
                   JOIN interactions i ON a.id = i.article_id
                   WHERE i.score = ?
                   ORDER BY i.created_at DESC""",
                (score,),
            ).fetchall()
            articles = [dict(r) for r in rows]
            for a in articles:
                blob = a.get("embedding")
                if blob and isinstance(blob, bytes):
                    a["embedding"] = _deserialize_embedding(blob)
            return articles

    # --- stats ---

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about interactions and personalisation data."""
        with self._conn() as conn:
            total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            new_articles = conn.execute(
                """SELECT COUNT(*) FROM articles a
                   LEFT JOIN interactions i ON a.id = i.article_id
                   WHERE i.article_id IS NULL"""
            ).fetchone()[0]
            likes = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE score = 1"
            ).fetchone()[0]
            dislikes = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE score = -1"
            ).fetchone()[0]
            neutrals = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE score = 0"
            ).fetchone()[0]

            top_sources = [
                dict(r)
                for r in conn.execute(
                    """SELECT s.name, ss.score, ss.likes, ss.dislikes
                       FROM source_scores ss
                       JOIN sources s ON ss.source_id = s.id
                       ORDER BY ss.score DESC
                       LIMIT 10"""
                ).fetchall()
            ]

            top_authors = [
                dict(r)
                for r in conn.execute(
                    """SELECT as2.author, as2.score, as2.likes, as2.dislikes, s.name as source_name
                       FROM author_scores as2
                       JOIN sources s ON as2.source_id = s.id
                       ORDER BY as2.score DESC
                       LIMIT 10"""
                ).fetchall()
            ]

            bottom_sources = [
                dict(r)
                for r in conn.execute(
                    """SELECT s.name, ss.score, ss.likes, ss.dislikes
                       FROM source_scores ss
                       JOIN sources s ON ss.source_id = s.id
                       ORDER BY ss.score ASC
                       LIMIT 5"""
                ).fetchall()
            ]

            return {
                "articles": {"total": total_articles, "new": new_articles},
                "interactions": {
                    "total": likes + dislikes + neutrals,
                    "likes": likes,
                    "dislikes": dislikes,
                    "neutrals": neutrals,
                },
                "top_sources": top_sources,
                "bottom_sources": bottom_sources,
                "top_authors": top_authors,
            }
