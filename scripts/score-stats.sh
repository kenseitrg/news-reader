#!/usr/bin/env bash
# Quick score-distribution diagnostics for news_reader.db
# Usage: ./scripts/score-stats.sh [path/to/news_reader.db]

set -euo pipefail

DB="${1:-data/news_reader.db}"

if [ ! -f "$DB" ]; then
    echo "Database not found: $DB"
    exit 1
fi

echo "=== Interaction distribution ==="
sqlite3 "$DB" "
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END) AS likes,
    SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) AS dislikes,
    SUM(CASE WHEN score = 0 THEN 1 ELSE 0 END) AS neutrals,
    ROUND(100.0 * SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS like_pct
FROM interactions;
"

echo ""
echo "=== Source scores ==="
sqlite3 -header "$DB" "
SELECT s.name, ss.score, ss.likes, ss.dislikes
FROM source_scores ss
JOIN sources s ON ss.source_id = s.id
ORDER BY ss.score DESC;
"

echo ""
echo "=== Top 15 authors by score ==="
sqlite3 -header "$DB" "
SELECT a2.author, s.name AS source, a2.score, a2.likes, a2.dislikes
FROM author_scores a2
JOIN sources s ON a2.source_id = s.id
WHERE a2.likes + a2.dislikes > 0
ORDER BY a2.score DESC
LIMIT 15;
"

echo ""
echo "=== Bottom 10 authors by score ==="
sqlite3 -header "$DB" "
SELECT a2.author, s.name AS source, a2.score, a2.likes, a2.dislikes
FROM author_scores a2
JOIN sources s ON a2.source_id = s.id
WHERE a2.likes + a2.dislikes > 0
ORDER BY a2.score ASC
LIMIT 10;
"

echo ""
echo "=== New articles (no interaction) ==="
sqlite3 "$DB" "
SELECT COUNT(*) AS new_articles
FROM articles a
LEFT JOIN interactions i ON a.id = i.article_id
WHERE i.article_id IS NULL;
"

echo ""
echo "=== Articles missing embeddings ==="
sqlite3 "$DB" "
SELECT COUNT(*) AS missing_embeddings
FROM articles WHERE embedding IS NULL;
"
