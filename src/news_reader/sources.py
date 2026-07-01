from dataclasses import dataclass


@dataclass
class Source:
    id: int | None
    name: str
    type: str  # "rss" | "scrape"
    url: str
    feed_url: str | None
    enabled: bool = True


def source_from_row(row: dict) -> Source:
    return Source(
        id=row["id"],
        name=row["name"],
        type=row["type"],
        url=row["url"],
        feed_url=row.get("feed_url"),
        enabled=bool(row["enabled"]),
    )
