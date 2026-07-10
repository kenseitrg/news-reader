"""Simple HTTP server providing a web UI for browsing and interacting with articles.

Usage:
    python -m news_reader.web
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import ParseResult, parse_qs, urlparse

from news_reader.config import load as load_config
from news_reader.ranker import Ranker
from news_reader.storage import Storage

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>News Reader</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #222; padding: 20px; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.5rem; margin-bottom: 16px; }
  .controls { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }
  .controls input { width: 100px; padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
  .controls button { padding: 6px 16px; border: none; border-radius: 4px; background: #007bff; color: #fff; font-size: 14px; cursor: pointer; }
  .controls button:disabled { opacity: .6; cursor: default; }
  .controls button:hover:not(:disabled) { background: #0056b3; }
  .controls .count { font-size: 13px; color: #666; }
  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }
  th { background: #f8f9fa; font-weight: 600; color: #555; }
  td.actions { white-space: nowrap; }
  td.actions button { padding: 4px 10px; border: 1px solid #ddd; border-radius: 3px; cursor: pointer; font-size: 13px; margin-right: 4px; background: #fff; }
  td.actions button:hover { opacity: .8; }
  td.actions .like:hover { background: #d4edda; border-color: #28a745; color: #155724; }
  td.actions .dislike:hover { background: #f8d7da; border-color: #dc3545; color: #721c24; }
  td.actions .read:hover { background: #cce5ff; border-color: #007bff; color: #004085; }
  td.actions button:disabled { opacity: .4; cursor: default; }
  td.actions button:disabled:hover { background: #fff; color: inherit; }
  .summary { max-width: 350px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #555; }
  .title-cell { max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .title-cell a { color: #007bff; text-decoration: none; }
  .title-cell a:hover { text-decoration: underline; }
  .score { font-variant-numeric: tabular-nums; }
  .error { color: #dc3545; padding: 12px; background: #f8d7da; border-radius: 4px; margin-bottom: 12px; }
  .empty { text-align: center; padding: 40px; color: #888; }
</style>
</head>
<body>
<div class="container">
  <h1>News Reader</h1>
  <div class="controls">
    <label for="limit">Articles:</label>
    <input type="number" id="limit" value="20" min="1" max="200">
    <button id="list-btn">List</button>
    <span class="count" id="count"></span>
  </div>
  <div id="error" class="error" style="display:none"></div>
  <table id="articles">
    <thead>
      <tr><th>ID</th><th>Title</th><th>Summary</th><th>Score</th><th>Actions</th></tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<script>
const tbody = document.getElementById('tbody');
const errorDiv = document.getElementById('error');
const countSpan = document.getElementById('count');
const listBtn = document.getElementById('list-btn');

listBtn.addEventListener('click', loadArticles);
document.getElementById('limit').addEventListener('keydown', e => { if (e.key === 'Enter') loadArticles(); });

async function loadArticles() {
  const limit = document.getElementById('limit').value || 20;
  listBtn.disabled = true;
  hideError();
  try {
    const res = await fetch(`/api/articles?limit=${limit}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderTable(data.articles);
  } catch (e) {
    showError(String(e));
  } finally {
    listBtn.disabled = false;
  }
}

function renderTable(articles) {
  countSpan.textContent = articles.length + ' articles';
  if (articles.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">No new articles</td></tr>';
    return;
  }
  tbody.innerHTML = articles.map(a => {
    const title = escapeHtml(a.title);
    const summary = escapeHtml((a.summary || '').substring(0, 120));
    const link = escapeHtml(a.link);
    return '<tr>' +
      `<td>${a.id}</td>` +
      `<td class="title-cell"><a href="${link}" target="_blank" title="${title}">${title}</a></td>` +
      `<td class="summary" title="${summary}">${summary}</td>` +
      `<td class="score">${a.score.toFixed(4)}</td>` +
      `<td class="actions">` +
        `<button class="like" onclick="interact(${a.id}, 1, this)">Like</button>` +
        `<button class="dislike" onclick="interact(${a.id}, -1, this)">Dislike</button>` +
        `<button class="read" onclick="interact(${a.id}, 0, this)">Read</button>` +
      `</td>` +
    '</tr>';
  }).join('');
}

async function interact(id, score, btn) {
  const row = btn.closest('tr');
  const buttons = row.querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);
  hideError();
  try {
    const res = await fetch(`/api/articles/${id}/interact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ score }),
    });
    if (!res.ok) throw new Error(await res.text());
    row.remove();
    const remaining = tbody.querySelectorAll('tr').length;
    countSpan.textContent = remaining + ' articles';
    if (remaining === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No new articles</td></tr>';
    }
  } catch (e) {
    buttons.forEach(b => b.disabled = false);
    showError(String(e));
  }
}

function showError(msg) { errorDiv.textContent = msg; errorDiv.style.display = ''; }
function hideError() { errorDiv.style.display = 'none'; }
function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>"""


_ARTICLE_ID_RE = re.compile(r"^/api/articles/(\d+)/interact$")


def _json_response(
    handler: BaseHTTPRequestHandler,
    data: Any,
    status: int = 200,
) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> bytes:
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return b""
    return handler.rfile.read(length)


class NewsReaderHandler(BaseHTTPRequestHandler):
    """HTTP request handler serving the web UI and REST API."""

    storage: Storage | None = None
    config: dict[str, Any] | None = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/api/articles":
            self._list_articles(parsed)
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_POST(self) -> None:
        match = _ARTICLE_ID_RE.match(self.path)
        if not match:
            _json_response(self, {"error": "Not found"}, 404)
            return

        article_id = int(match.group(1))
        try:
            body = json.loads(_read_body(self))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _json_response(self, {"error": "Invalid JSON body"}, 400)
            return

        score = body.get("score")
        if score not in (-1, 0, 1):
            _json_response(self, {"error": "score must be -1, 0, or 1"}, 400)
            return

        assert self.storage is not None
        self.storage.set_interaction(article_id, score)
        _json_response(self, {"ok": True})

    def _serve_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _list_articles(self, parsed: ParseResult) -> None:
        params = parse_qs(parsed.query)
        limit = int(params.get("limit", ["20"])[0])

        assert self.storage is not None
        assert self.config is not None

        articles = self.storage.get_new_articles()

        liked = self.storage.get_interacted_articles(score=1)
        disliked = self.storage.get_interacted_articles(score=-1)

        liked_embs = [a["embedding"] for a in liked if a.get("embedding")]
        disliked_embs = [a["embedding"] for a in disliked if a.get("embedding")]

        source_scores = self.storage.get_source_scores()
        author_scores = self.storage.get_author_scores()

        ranker = Ranker(self.config)
        ranker.score_articles(
            articles,
            liked_embs,
            disliked_embs,
            source_scores=source_scores,
            author_scores=author_scores,
        )

        result = []
        for a in articles[:limit]:
            result.append(
                {
                    "id": a["id"],
                    "title": a["title"],
                    "summary": a.get("summary", "") or "",
                    "score": a.get("_score", 0.0),
                    "link": a["link"],
                    "published_at": a.get("published_at", "") or "",
                }
            )

        _json_response(self, {"articles": result})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the web server.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
    """
    config = load_config()
    storage = Storage(config["db_path"])

    NewsReaderHandler.storage = storage
    NewsReaderHandler.config = config

    server = HTTPServer((host, port), NewsReaderHandler)
    print(f"News Reader UI — http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    serve()
