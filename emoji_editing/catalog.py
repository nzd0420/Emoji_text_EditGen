"""Catalog helpers for browsing the curated emoji editing dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmojiCatalogEntry:
    """Single emoji option shown in the interactive UI."""

    key: str
    row_id: int
    emoji: str
    name: str
    vendor: str
    unicode_slug: str
    image_path: str

    @property
    def display_name(self) -> str:
        return f"{self.emoji} {self.name}"


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_vendor_catalog(vendor_index_csv: str | Path) -> list[EmojiCatalogEntry]:
    """Load the processed vendor image index into UI-friendly entries."""

    rows = _read_rows(vendor_index_csv)
    entries = [
        EmojiCatalogEntry(
            key=f"{row['vendor']}::{row['row_id']}",
            row_id=int(row["row_id"]),
            emoji=row["emoji"],
            name=row["name"],
            vendor=row["vendor"],
            unicode_slug=row["unicode_slug"],
            image_path=row["processed_image_path"],
        )
        for row in rows
    ]
    entries.sort(key=lambda item: (item.vendor, item.row_id))
    return entries


def vendors_from_catalog(entries: list[EmojiCatalogEntry]) -> list[str]:
    return sorted({entry.vendor for entry in entries})


def entries_for_vendor(entries: list[EmojiCatalogEntry], vendor: str) -> list[EmojiCatalogEntry]:
    return [entry for entry in entries if entry.vendor == vendor]


def catalog_lookup(entries: list[EmojiCatalogEntry]) -> dict[str, EmojiCatalogEntry]:
    return {entry.key: entry for entry in entries}
