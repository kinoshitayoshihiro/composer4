#!/usr/bin/env python3
"""Rhythm vocabulary manifest loader for RhythmAI/Rulebook/EmotionAI."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

LOGGER = logging.getLogger(__name__)
_DEFAULT_MANIFEST = Path("data/rhythm_vocab.yaml")


@dataclass(frozen=True)
class LibrarySpec:
    name: str
    instrument: str
    path: Path
    root: Optional[str]
    format: Optional[str] = None
    notes: Optional[str] = None

    def root_parts(self) -> List[str]:
        if not self.root:
            return []
        return [part for part in self.root.split(".") if part]


@dataclass(frozen=True)
class PatternEntry:
    id: str
    instrument: str
    source: str
    pattern_ref: str
    density: Optional[str] = None
    energy: Optional[str] = None
    grid: Optional[float] = None
    sections: List[str] = field(default_factory=list)
    descriptors: List[str] = field(default_factory=list)
    ai_hooks: Dict[str, Any] = field(default_factory=dict)

    def ref_parts(self) -> List[str]:
        return [part for part in self.pattern_ref.split(".") if part]


@dataclass(frozen=True)
class EnsembleSpec:
    id: str
    description: str
    layers: Dict[str, str]


@dataclass
class RhythmVocab:
    metadata: Dict[str, Any]
    schema: Dict[str, Any]
    libraries: Dict[str, LibrarySpec]
    vocabulary: Dict[str, List[PatternEntry]]
    ensembles: List[EnsembleSpec]

    def entries(self) -> Iterable[PatternEntry]:
        for entries in self.vocabulary.values():
            for entry in entries:
                yield entry

    def get_entry(self, entry_id: str) -> PatternEntry:
        for entry in self.entries():
            if entry.id == entry_id:
                return entry
        raise KeyError(f"Pattern id not found: {entry_id}")

    def get_library(self, name: str) -> LibrarySpec:
        try:
            return self.libraries[name]
        except KeyError as exc:
            raise KeyError(f"Library '{name}' is not declared in manifest") from exc


class RhythmVocabLoader:
    """Loads rhythm_vocab.yaml and resolves pattern references on demand."""

    def __init__(self, manifest_path: Path | None = None, *, validate: bool = False):
        self.manifest_path = (manifest_path or _DEFAULT_MANIFEST).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Rhythm vocab manifest not found: {self.manifest_path}")

        LOGGER.debug("Loading rhythm vocab manifest: %s", self.manifest_path)
        with self.manifest_path.open("r", encoding="utf-8") as fp:
            doc = yaml.safe_load(fp)

        libraries = {
            name: LibrarySpec(
                name=name,
                instrument=spec.get("instrument", "unknown"),
                path=(self.manifest_path.parent / spec["path"]).resolve(),
                root=spec.get("root"),
                format=spec.get("format"),
                notes=spec.get("notes"),
            )
            for name, spec in (doc.get("libraries") or {}).items()
        }

        vocabulary: Dict[str, List[PatternEntry]] = {}
        for instrument, entries in (doc.get("vocabulary") or {}).items():
            vocabulary[instrument] = [
                PatternEntry(
                    id=item["id"],
                    instrument=item.get("instrument", instrument),
                    source=item["source"],
                    pattern_ref=item["pattern_ref"],
                    density=item.get("density"),
                    energy=item.get("energy"),
                    grid=item.get("grid"),
                    sections=list(item.get("sections", [])),
                    descriptors=list(item.get("descriptors", [])),
                    ai_hooks=dict(item.get("ai_hooks", {})),
                )
                for item in entries or []
            ]

        ensembles = [
            EnsembleSpec(
                id=item["id"],
                description=item.get("description", ""),
                layers=dict(item.get("layers", {})),
            )
            for item in (doc.get("ensembles") or [])
        ]

        self.vocab = RhythmVocab(
            metadata=doc.get("metadata", {}),
            schema=doc.get("schema", {}),
            libraries=libraries,
            vocabulary=vocabulary,
            ensembles=ensembles,
        )
        self._cache: Dict[Path, Dict[str, Any]] = {}

        if validate:
            self.validate_entries()

    # ------------------------------------------------------------------
    def list_entries(self, instrument: Optional[str] = None) -> List[PatternEntry]:
        if instrument:
            return list(self.vocab.vocabulary.get(instrument, []))
        return list(self.vocab.entries())

    def list_ensembles(self) -> List[EnsembleSpec]:
        return list(self.vocab.ensembles)

    def resolve_pattern(self, entry_id: str) -> Dict[str, Any]:
        entry = self.vocab.get_entry(entry_id)
        library = self.vocab.get_library(entry.source)

        data = self._load_library_file(library.path)
        node: Any = data
        for part in library.root_parts():
            node = _index_node(node, part)
        for part in entry.ref_parts():
            node = _index_node(node, part)

        if not isinstance(node, dict):
            raise TypeError(
                f"Resolved pattern for '{entry_id}' is not a mapping (got {type(node).__name__})"
            )
        return node

    def resolve_ensemble(self, ensemble_id: str) -> EnsembleSpec:
        for ens in self.vocab.ensembles:
            if ens.id == ensemble_id:
                return ens
        raise KeyError(f"Ensemble id not found: {ensemble_id}")

    def validate_entries(self) -> None:
        errors: List[str] = []
        for entry in self.vocab.entries():
            try:
                self.resolve_pattern(entry.id)
            except Exception as exc:  # pragma: no cover - validation aid
                errors.append(f"{entry.id}: {exc}")
        if errors:
            raise ValueError(
                "Rhythm vocab validation failed:\n" + "\n".join(f" - {err}" for err in errors)
            )

    # ------------------------------------------------------------------
    def _load_library_file(self, path: Path) -> Dict[str, Any]:
        if path not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"Library file not found: {path}")
            with path.open("r", encoding="utf-8") as fp:
                if path.suffix.lower() == ".json":
                    data = json.load(fp)
                else:
                    data = yaml.safe_load(fp)
            if not isinstance(data, dict):
                raise TypeError(f"Library root must be a mapping (file: {path})")
            self._cache[path] = data
        return self._cache[path]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect rhythm vocab manifest")
    parser.add_argument(
        "manifest",
        nargs="?",
        default=str(_DEFAULT_MANIFEST),
        help="Path to rhythm_vocab.yaml (defaults to data/rhythm_vocab.yaml)",
    )
    parser.add_argument("--instrument", help="Filter listing to a specific instrument")
    parser.add_argument("--entry", help="Resolve a specific entry id and print the pattern")
    parser.add_argument("--ensemble", help="Show ensemble definition and referenced entries")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that every entry resolves to an underlying pattern",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of repr")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    return parser


def _main() -> None:  # pragma: no cover - CLI utility
    parser = _create_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    loader = RhythmVocabLoader(Path(args.manifest), validate=args.validate)

    if args.entry:
        pattern = loader.resolve_pattern(args.entry)
        if args.json:
            print(json.dumps(pattern, ensure_ascii=False, indent=2))
        else:
            print(pattern)
        return

    if args.ensemble:
        ens = loader.resolve_ensemble(args.ensemble)
        payload = {
            "id": ens.id,
            "description": ens.description,
            "layers": {
                role: loader.vocab.get_entry(pattern_id).pattern_ref
                for role, pattern_id in ens.layers.items()
            },
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload)
        return

    entries = loader.list_entries(args.instrument)
    payload = [entry.__dict__ for entry in entries]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for entry in entries:
            print(f"{entry.id} -> {entry.pattern_ref} (source={entry.source})")


if __name__ == "__main__":  # pragma: no cover
    _main()
