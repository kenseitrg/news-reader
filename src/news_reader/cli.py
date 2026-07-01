from typing import Annotated, Optional

import asyncio

import httpx
import typer
from rich.console import Console
from rich.table import Table

from news_reader.config import load as load_config
from news_reader.fetcher import fetch_rss, fetch_scrape
from news_reader.sources import source_from_row
from news_reader.storage import Storage

console = Console()
app = typer.Typer()


@app.command()
def fetch() -> None:
    """Fetch new articles from all enabled sources."""
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

                console.print(f"  {source.name}: {len(articles)} found, {new_count} new")
                total += new_count
        return total

    total_new = asyncio.run(_fetch_all())
    console.print(f"[green]Done. {total_new} new articles.[/green]")


@app.command(name="list")
def list_(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max articles to show")] = 20,
) -> None:
    """Show unread articles."""
    config = load_config()
    storage = Storage(config["db_path"])

    articles = storage.get_uninteracted_articles()
    if not articles:
        console.print("[green]No new articles.[/green]")
        raise typer.Exit()

    table = Table(title="Articles")
    table.add_column("#", style="dim")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Published")

    for i, article in enumerate(articles[:limit], 1):
        table.add_row(
            str(i),
            article.get("source_name", ""),
            article["title"][:80],
            (article.get("published_at") or "")[:10],
        )
    console.print(table)


@app.command()
def like(article_id: Annotated[int, typer.Argument(help="Article ID")]) -> None:
    """Mark article as liked."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.set_interaction(article_id, liked=True)
    console.print(f"[green]Liked article {article_id}[/green]")


@app.command()
def dislike(article_id: Annotated[int, typer.Argument(help="Article ID")]) -> None:
    """Mark article as disliked."""
    config = load_config()
    storage = Storage(config["db_path"])
    storage.set_interaction(article_id, liked=False)
    console.print(f"[yellow]Disliked article {article_id}[/yellow]")


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
    type_: Annotated[str, typer.Option("--type", "-t", help="Source type (rss|scrape)")] = "rss",
    feed_url: Annotated[Optional[str], typer.Option("--feed-url", "-f", help="RSS feed URL if different")] = None,
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


if __name__ == "__main__":
    app()
