#!/usr/bin/env python3
"""Continue Module — extend motifs using RhythmAI + Stage3 metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from otobonAI.rhythm_ai import RhythmAI as RhythmAIType
else:
    RhythmAIType = Any  # type: ignore[misc]

try:
    from otobonAI.rhythm_ai import RhythmAI as RhythmAIRuntime
except Exception:  # pragma: no cover
    RhythmAIRuntime = None


@dataclass
class Stage3Condition:
    """Subset of Stage3 metrics used by Continue heuristics."""

    loop_id: str
    density: float
    intensity: float
    swing: float
    valence: float
    arousal: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "Stage3Condition":
        loop_id = str(row.get("loop_id") or row.get("file") or "unknown")
        density = _scaled_density(row.get("backbeat_strength"), row.get("n_downbeats"))
        intensity = _scaled_intensity(row.get("arousal"))
        swing = _scaled_swing(row.get("swing_pct"))
        valence = float(row.get("valence", 0.0) or 0.0)
        arousal = float(row.get("arousal", 0.0) or 0.0)
        return cls(
            loop_id=loop_id,
            density=density,
            intensity=intensity,
            swing=swing,
            valence=valence,
            arousal=arousal,
        )

    @classmethod
    def default(cls) -> "Stage3Condition":
        return cls(
            loop_id="default", density=1.0, intensity=1.0, swing=0.0, valence=0.0, arousal=0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "density": self.density,
            "intensity": self.intensity,
            "swing": self.swing,
            "valence": self.valence,
            "arousal": self.arousal,
        }


def _scaled_density(backbeat_strength: Any, n_downbeats: Any) -> float:
    if backbeat_strength is None or backbeat_strength != backbeat_strength:
        fallback = (float(n_downbeats) if n_downbeats else 16.0) / 16.0
    else:
        fallback = float(backbeat_strength)
    scaled = 0.85 + fallback * 4.5
    return float(np.clip(scaled, 0.5, 2.0))


def _scaled_intensity(arousal: Any) -> float:
    value = float(arousal) if arousal is not None else 0.0
    scaled = 1.0 + value * 0.6
    return float(np.clip(scaled, 0.4, 1.8))


def _scaled_swing(swing_pct: Any) -> float:
    value = float(swing_pct) if swing_pct is not None else 0.0
    return float(np.clip(value / 100.0, 0.0, 0.6))


class ContinueModule:
    """Motif continuation engine."""

    def __init__(
        self,
        *,
        rhythm_ai: Optional["RhythmAIType"] = None,
        stage3_df: Optional[pd.DataFrame] = None,
        stage3_loop_id: Optional[str] = None,
        beats_per_bar: float = 4.0,
        seed: int = 42,
    ) -> None:
        self.rhythm_ai = rhythm_ai
        self.stage3_df = stage3_df if stage3_df is not None else pd.DataFrame()
        self.stage3_loop_id = stage3_loop_id
        self.beats_per_bar = beats_per_bar
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._condition_cache: Dict[tuple[str, int], Stage3Condition] = {}

    def extend(
        self,
        motif_events: List[Dict[str, Any]],
        *,
        source_bars: int,
        target_bars: int,
        instrument: str,
        section_label: str,
    ) -> Dict[str, Any]:
        """Extend motif events to target bars."""
        if not motif_events:
            raise ValueError("motif events cannot be empty")
        if target_bars <= source_bars:
            raise ValueError("target_bars must be greater than source_bars")

        bar_len_ql = self.beats_per_bar
        motif_length = self._estimate_length(motif_events, source_bars, bar_len_ql)
        normalized = self._normalize_events(motif_events)
        extended: List[Dict[str, Any]] = []
        applied_conditions: List[Dict[str, Any]] = []

        extended.extend(self._render(normalized, offset_ql=0.0))

        current_bar = source_bars
        while current_bar < target_bars:
            cond = self._condition_for(section_label, current_bar)
            density_bucket = _density_bucket(cond.density)
            pattern_meta = self._choose_pattern(
                instrument=instrument,
                section_label=section_label,
                density_hint=density_bucket,
            )
            applied_conditions.append({"bar": current_bar, **cond.to_dict()})
            offset = current_bar * bar_len_ql
            replicated = self._render(
                normalized,
                offset_ql=offset,
                density_scale=cond.density,
                velocity_bias=int((cond.intensity - 1.0) * 28),
                swing_ratio=cond.swing,
                pattern_meta=pattern_meta,
            )
            extended.extend(replicated)
            current_bar += source_bars

        metadata: Dict[str, Any] = {
            "instrument": instrument,
            "section": section_label,
            "source_bars": source_bars,
            "target_bars": target_bars,
            "motif_length_ql": motif_length,
            "stage3_conditions": applied_conditions,
            "rhythm_pattern_ids": sorted(
                {
                    evt["rhythm_pattern_id"]
                    for evt in extended
                    if isinstance(evt.get("rhythm_pattern_id"), str)
                }
            ),
        }

        return {"events": sorted(extended, key=lambda evt: evt["time_ql"]), "metadata": metadata}

    # ------------------------------------------------------------------ helpers
    def _normalize_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        base = min(evt.get("time_ql", evt.get("start_ql", 0.0)) for evt in events)
        normalized: List[Dict[str, Any]] = []
        for evt in events:
            cloned = dict(evt)
            start = evt.get("time_ql", evt.get("start_ql", 0.0)) - base
            cloned["time_ql"] = float(start)
            cloned["duration_ql"] = float(evt.get("duration_ql", evt.get("dur", 1.0)))
            normalized.append(cloned)
        return normalized

    def _render(
        self,
        events: List[Dict[str, Any]],
        *,
        offset_ql: float,
        density_scale: float = 1.0,
        velocity_bias: int = 0,
        swing_ratio: float = 0.0,
        pattern_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        density_scale = float(np.clip(density_scale, 0.5, 2.0))
        swing_ratio = float(np.clip(swing_ratio, -0.2, 0.25))
        drop_prob = 0.0
        multiplier = 1
        if density_scale > 1.35:
            multiplier = 2
        elif density_scale < 0.85:
            drop_prob = 0.2 + (0.85 - density_scale)

        rendered: List[Dict[str, Any]] = []
        for evt in events:
            if drop_prob and self.rng.random() < drop_prob:
                continue
            for copy_idx in range(multiplier):
                start = evt["time_ql"] + offset_ql + copy_idx * 0.125
                if swing_ratio and ((start / 0.5) % 2) >= 1:
                    start += swing_ratio * 0.5
                duration = max(
                    0.125, evt.get("duration_ql", 1.0) * (1.0 + (density_scale - 1.0) * 0.2)
                )
                velocity = int(np.clip(evt.get("velocity", 80) + velocity_bias, 12, 120))

                cloned = dict(evt)
                cloned["time_ql"] = float(start)
                cloned["duration_ql"] = float(duration)
                cloned["bar_idx"] = int(start // self.beats_per_bar)
                cloned["velocity"] = velocity
                if pattern_meta:
                    cloned.update(pattern_meta)
                rendered.append(cloned)
        return rendered

    @staticmethod
    def _estimate_length(
        events: List[Dict[str, Any]], source_bars: int, bar_len_ql: float
    ) -> float:
        starts = [evt.get("time_ql", evt.get("start_ql", 0.0)) for evt in events]
        ends = [
            start + evt.get("duration_ql", evt.get("dur", 1.0))
            for start, evt in zip(starts, events)
        ]
        span = max(ends) - min(starts)
        min_span = source_bars * bar_len_ql
        return float(max(span, min_span))

    def _condition_for(self, section_label: str, bar_index: int) -> Stage3Condition:
        key = (section_label, bar_index)
        if key in self._condition_cache:
            return self._condition_cache[key]

        if self.stage3_df.empty:
            cond = Stage3Condition.default()
        else:
            subset = self.stage3_df
            if self.stage3_loop_id and "loop_id" in subset.columns:
                narrowed = subset[subset["loop_id"] == self.stage3_loop_id]
                if not narrowed.empty:
                    subset = narrowed
            row = subset.iloc[int(self.rng.integers(0, len(subset)))]
            cond = Stage3Condition.from_row(row)

        self._condition_cache[key] = cond
        return cond

    def _choose_pattern(
        self,
        *,
        instrument: str,
        section_label: str,
        density_hint: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.rhythm_ai or not getattr(self.rhythm_ai, "has_manifest", lambda: False)():
            return None
        entry = self.rhythm_ai.choose_vocab_entry(
            instrument=instrument,
            section_label=section_label,
            density_hint=density_hint,
        )
        if not entry:
            return None
        return {
            "rhythm_pattern_id": entry.id,
            "pattern_source": entry.source,
            "pattern_ref": entry.pattern_ref,
        }


def _density_bucket(value: float) -> str:
    if value <= 0.75:
        return "sparse"
    if value <= 1.1:
        return "medium"
    if value <= 1.5:
        return "dense"
    return "wall"


def load_events(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "events" in payload:
        return list(payload["events"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("motif JSON must be a list or contain an 'events' key")


def load_stage3(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"Stage3 conditions not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_rhythm_ai(
    groove_vocab: Optional[Path],
    manifest: Optional[Path],
) -> Optional["RhythmAIType"]:
    if RhythmAIRuntime is None:
        return None
    try:
        return RhythmAIRuntime(
            vocab_path=groove_vocab if groove_vocab else None,
            rhythm_manifest_path=manifest if manifest else None,
            enable_logging=False,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[ContinueModule] RhythmAI unavailable: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend motifs via Continue module")
    parser.add_argument("--motif", required=True, help="Path to motif JSON (list or {events})")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--source-bars", type=int, default=1, help="Bars covered by the motif")
    parser.add_argument("--target-bars", type=int, default=8, help="Bars after extension")
    parser.add_argument("--instrument", default="piano", help="Instrument tag for RhythmAI")
    parser.add_argument("--section", default="verse", help="Section label")
    parser.add_argument("--beats-per-bar", type=float, default=4.0, help="Beats (quarters) per bar")
    parser.add_argument("--stage3-conditions", help="Stage3 conditions parquet/csv path")
    parser.add_argument("--stage3-loop-id", help="Optional loop_id to anchor Stage3 stats")
    parser.add_argument("--groove-vocab", help="Groove vocab parquet path")
    parser.add_argument("--rhythm-manifest", help="rhythm_vocab.yaml manifest path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motif_path = Path(args.motif)
    out_path = Path(args.out)
    stage3_path = Path(args.stage3_conditions).expanduser() if args.stage3_conditions else None
    groove_path = Path(args.groove_vocab).expanduser() if args.groove_vocab else None
    manifest_path = Path(args.rhythm_manifest).expanduser() if args.rhythm_manifest else None

    events = load_events(motif_path)
    stage3_df = load_stage3(stage3_path)
    rhythm_ai = build_rhythm_ai(groove_path, manifest_path)

    module = ContinueModule(
        rhythm_ai=rhythm_ai,
        stage3_df=stage3_df,
        stage3_loop_id=args.stage3_loop_id,
        beats_per_bar=args.beats_per_bar,
        seed=args.seed,
    )
    result = module.extend(
        events,
        source_bars=args.source_bars,
        target_bars=args.target_bars,
        instrument=args.instrument,
        section_label=args.section,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Continue module wrote {len(result['events'])} events -> {out_path}")


if __name__ == "__main__":
    main()
