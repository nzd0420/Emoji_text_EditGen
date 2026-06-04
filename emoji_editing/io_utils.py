"""Shared lightweight IO helpers (stdlib only, no torch/PIL imports)."""

from __future__ import annotations

import csv
from pathlib import Path


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts."""

    path = Path(csv_path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
