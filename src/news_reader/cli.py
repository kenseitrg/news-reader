from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from news_reader.config import load as load_config
from news_reader.sources import source_from_row
from news_reader.storage import Storage

console = Console()
app = typer.Typer()


@app.command()
def fetch() -> None:
    """Fetch new articles from all enabled sources."""
    import asyncio

    import httpx

    from news_reader.fetcher import fetch_rss, fetch_scrape

    config = load_config()
    storage = Storage(config["db_path"])
    sources = [source_from_row(s) for s in storage.get_sources(enabled_only=True)]

    if not sources:
        console.print("[yellow]No enabled sources. Add one first.[/yellow]")
        raise typer.Exit()

    async def _fetch_all() -> int:
        total = 0
        async with httpx.AsyncClient() as client:
            for source in sources:
                if source.type == "rss":
                    articles = await fetch_rss(source, client)
                else:
                    articles = await fetch_scrape(source, client)

                new_count = 0
                for article in articles:
                    if storage.add_article(article):
                        new_count += 1

                console.print(
                    f"  {source.name}: {len(articles)} found, {new_count} new"
                )
                total += new_count
        return total

    total_new = asyncio.run(_fetch_all())

    if total_new > 0:
        console.print()
        summarized = _summarize_new(storage, config)
        console.print(f"[green]Summarized {summarized} articles.[/green]")

    console.print(f"[green]Done. {total_new} new articles.[/green]")


@app.command(name="list")
def list_(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Max articles to show")
    ] = 20,
    fresh: Annotated[
        bool, typer.Option("--fresh", "-f", help="Sort by date instead of relevance")
    ] = False,
) -> None:
    """Show unread articles, ranked by relevance."""
    config = load_config()
    storage = Storage(config["db_path"])

    articles = storage.get_new_articles()
    if not articles:
        console.print("[green]No new articles.[/green]")
        raise typer.Exit()

    from news_reader.ranker import Ranker

    liked = storage.get_interacted_articles(score=1)
    disliked = storage.get_interacted_articles(score=-1)

    liked_embs = [a["embedding"] for a in liked if a.get("embedding")]
    disliked_embs = [a["embedding"] for a in disliked if a.get("embedding")]

    source_scores = storage.get_source_scores()
    author_scores = storage.get_author_scores()

    ranker = Ranker(config)
    ranker.score_articles(
        articles,
        liked_embs,
        disliked_embs,
        source_scores=source_scores,
        author_scores=author_scores,
    )

    if fresh:
        articles.sort(key=lambda a: a.get("published_at") or "", reverse=True)

    table = Table(title="Articles")
    table.add_column("ID", style="dim")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Summary")
    table.add_column("Score")
    table.add_column("Published")

    for article in articles[:limit]:
        summary = (article.get("summary") or "")[:60]
        title_link = Text(article["title"][:80], style=f"link {article['link']}")
        row = [
            str(article["id"]),
            article.get("source_name", ""),
            title_link,
            summary,
            f"{article.get('_score', 0):.3f}",
            (article.get("published_at") or "")[:10],
        ]
        table.add_row(*row)
    console.print(table)


@app.command()
def like(article_id: Annotated[int, typer.Argument(help="Article ID")]) -> None:
    """Mark article as liked."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.set_interaction(article_id, 1)
    console.print(f"[green]Liked article {article_id}[/green]")


@app.command()
def dislike(article_id: Annotated[int, typer.Argument(help="Article ID")]) -> None:
    """Mark article as disliked."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.set_interaction(article_id, -1)
    console.print(f"[yellow]Disliked article {article_id}[/yellow]")


@app.command()
def read(article_id: Annotated[int, typer.Argument(help="Article ID")]) -> None:
    """Mark article as read (no opinion)."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.set_interaction(article_id, 0)
    console.print(f"[blue]Marked article {article_id} as read[/blue]")


@app.command()
def sources() -> None:
    """List all sources."""
    config = load_config()
    storage = Storage(config["db_path"])
    rows = storage.get_sources(enabled_only=False)

    table = Table(title="Sources")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("URL")
    table.add_column("Enabled")

    for row in rows:
        table.add_row(
            str(row["id"]),
            row["name"],
            row["type"],
            row["url"][:50],
            "✓" if row["enabled"] else "✗",
        )
    console.print(table)


@app.command()
def source_add(
    name: Annotated[str, typer.Option("--name", "-n", help="Source name")],
    url: Annotated[str, typer.Option("--url", "-u", help="Source URL")],
    type_: Annotated[
        str, typer.Option("--type", "-t", help="Source type (rss|scrape)")
    ] = "rss",
    feed_url: Annotated[
        Optional[str],
        typer.Option("--feed-url", "-f", help="RSS feed URL if different"),
    ] = None,
) -> None:
    """Add a new source."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.add_source(name, type_, url, feed_url)
    console.print(f"[green]Added source '{name}'[/green]")


@app.command()
def source_remove(
    source_id: Annotated[int, typer.Argument(help="Source ID")],
) -> None:
    """Remove a source."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.remove_source(source_id)
    console.print(f"[green]Removed source {source_id}[/green]")


@app.command()
def web() -> None:
    """Launch the web UI."""
    from news_reader.web import serve

    serve()


def _summarize_new(storage: Storage, config: dict, force: bool = False) -> int:
    """Summarize articles that don't have a meaningful summary yet."""
    import asyncio

    import httpx

    from news_reader.embeddings import Embedder
    from news_reader.fetcher import fetch_article_content
    from news_reader.summarizer import Summarizer

    to_summarize = storage.get_articles_without_summaries(force=force)
    if not to_summarize:
        return 0

    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path", "models/model.gguf")
    summarizer_cfg = config.get("summarizer", {})

    try:
        summarizer = Summarizer(
            model_path=model_path,
            n_ctx=model_cfg.get("n_ctx", 2048),
            n_threads=model_cfg.get("n_threads"),
            max_tokens=summarizer_cfg.get("max_tokens", 256),
            temperature=summarizer_cfg.get("temperature", 0.3),
        )
    except Exception as exc:
        console.print(f"[red]Failed to load model: {exc}[/red]")
        return 0

    embedder = Embedder()

    async def _fetch_and_summarize() -> int:
        done = 0
        async with httpx.AsyncClient() as client:
            for article in to_summarize:
                console.print(f"  Fetching content: {article['title'][:60]}...")
                content = await fetch_article_content(article["link"], client)
                if not content:
                    console.print("  (no content, falling back to title)")
                    content = article["title"]

                console.print("  Summarizing...")
                summary = summarizer.summarize(content)
                if summary:
                    storage.update_article_summary(article["id"], summary, None)

                    console.print("  Computing embedding...")
                    emb = embedder.embed_article(
                        title=article["title"],
                        summary=summary,
                    )
                    storage.update_article_embedding(article["id"], emb)
                    done += 1
        return done

    return asyncio.run(_fetch_and_summarize())


@app.command()
def summarize(
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Re-summarize all articles")
    ] = False,
) -> None:
    """Summarize articles without a meaningful summary."""
    config = load_config()
    storage = Storage(config["db_path"])
    count = _summarize_new(storage, config, force=force)
    console.print(f"[green]Summarized {count} articles.[/green]")


@app.command()
def embed() -> None:
    """Compute embeddings for articles that don't have one yet."""
    from news_reader.embeddings import Embedder

    config = load_config()
    storage = Storage(config["db_path"])
    missing = storage.get_articles_without_embeddings()

    if not missing:
        console.print("[green]All articles already have embeddings.[/green]")
        raise typer.Exit()

    embedder = Embedder()
    for article in missing:
        text = article["title"]
        summary = article.get("summary")
        if summary:
            text = f"{text}\n\n{summary}"
        console.print(f"  Embedding: {article['title'][:60]}...")
        emb = embedder.embed(text)
        storage.update_article_embedding(article["id"], emb)

    console.print(f"[green]Embedded {len(missing)} articles.[/green]")


@app.command()
def stats() -> None:
    """Show personalisation statistics."""
    config = load_config()
    storage = Storage(config["db_path"])
    data = storage.get_stats()

    i = data["interactions"]
    total = i["total"]
    likes = i["likes"]
    dislikes = i["dislikes"]
    neutrals = i["neutrals"]

    console.print()
    console.print("[bold]Overview[/bold]")
    summary = Table(show_header=False, box=None)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("Total articles", str(data["articles"]["total"]))
    summary.add_row("New (unranked)", str(data["articles"]["new"]))
    summary.add_row("Total interactions", str(total))
    summary.add_row("  👍 Liked", str(likes))
    summary.add_row("  👎 Disliked", str(dislikes))
    summary.add_row("  🤷 Neutral", str(neutrals))
    if total:
        summary.add_row("  Like ratio", f"{100.0 * likes / total:.1f}%")
    console.print(summary)

    if data["top_sources"]:
        console.print()
        console.print("[bold]Top Sources[/bold]")
        tsrc = Table()
        tsrc.add_column("Source")
        tsrc.add_column("Score")
        tsrc.add_column("Likes")
        tsrc.add_column("Dislikes")
        for row in data["top_sources"]:
            raw = row["score"]
            tsrc.add_row(
                row["name"],
                f"{raw:+.4f}",
                str(row["likes"]),
                str(row["dislikes"]),
            )
        console.print(tsrc)

    if data["bottom_sources"]:
        console.print()
        console.print("[bold]Worst Sources[/bold]")
        bsrc = Table()
        bsrc.add_column("Source")
        bsrc.add_column("Score")
        bsrc.add_column("Likes")
        bsrc.add_column("Dislikes")
        for row in data["bottom_sources"]:
            bsrc.add_row(
                row["name"],
                f"{row['score']:+.4f}",
                str(row["likes"]),
                str(row["dislikes"]),
            )
        console.print(bsrc)

    if data["top_authors"]:
        console.print()
        console.print("[bold]Top Authors[/bold]")
        tauth = Table()
        tauth.add_column("Author")
        tauth.add_column("Source")
        tauth.add_column("Score")
        tauth.add_column("Likes")
        tauth.add_column("Dislikes")
        for row in data["top_authors"]:
            tauth.add_row(
                row["author"],
                row["source_name"],
                f"{row['score']:+.4f}",
                str(row["likes"]),
                str(row["dislikes"]),
            )
        console.print(tauth)


if __name__ == "__main__":
    app()
