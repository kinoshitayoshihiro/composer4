from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

from .base_adapter import DatasetAdapter, DatasetItem


@dataclass
class Pop909Adapter(DatasetAdapter):
    """Adapter for the POP909 dataset using the common ingestion contract."""

    root: Path
    metadata_csv: Optional[Path] = None
    name: str = "pop909"

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        if self.metadata_csv is None:
            candidate = self.root / "metadata.csv"
            self.metadata_csv = candidate if candidate.exists() else None
        elif not self.metadata_csv.is_absolute():
            self.metadata_csv = (self.root / self.metadata_csv).resolve()

        self._metadata = self._load_metadata()

    def iter_items(self) -> Iterator[DatasetItem]:
        for midi_path in sorted(self.root.glob("**/*.mid")):
            loop_id = midi_path.stem
            record = self._metadata.get(loop_id, {})
            meta: Dict[str, object] = {
                "emotion": record.get("emotion"),
                "genre": record.get("genre"),
                "tempo_bpm": self._safe_float(record.get("tempo_bpm")),
                "key": record.get("key"),
            }
            license_origin = record.get("license_origin", "research_only")
            yield DatasetItem(
                loop_id=loop_id,
                midi_path=midi_path,
                meta=meta,
                license_origin=str(license_origin),
            )

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "root": str(self.root),
            "metadata_rows": len(self._metadata),
        }

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        if not self.metadata_csv or not self.metadata_csv.exists():
            return {}
        mapping: Dict[str, Dict[str, str]] = {}
        with self.metadata_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("filename") or row.get("loop_id")
                if not filename:
                    continue
                loop_id = Path(filename).stem
                mapping[loop_id] = row
        return mapping

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
