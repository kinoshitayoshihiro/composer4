from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class DatasetItem:
    """Normalized representation of a single ingestible musical asset."""

    loop_id: str
    midi_path: Path
    meta: Dict[str, object]
    license_origin: str


@runtime_checkable
class DatasetAdapter(Protocol):
    """Protocol for pluggable dataset ingestion adapters."""

    name: str

    def iter_items(self) -> Iterator[DatasetItem]:
        """Yield normalized dataset items for ingestion."""
        raise NotImplementedError

    def summary(self) -> Dict[str, object]:
        """Return quick adapter stats (counts, missing metadata, etc.)."""
        raise NotImplementedError


class InMemoryAdapter:
    """Simple adapter for tests and small curated collections."""

    def __init__(self, name: str, items: Iterable[DatasetItem]) -> None:
        self.name = name
        self._items = list(items)

    def iter_items(self) -> Iterator[DatasetItem]:
        yield from self._items

    def summary(self) -> Dict[str, object]:
        return {"count": len(self._items)}
