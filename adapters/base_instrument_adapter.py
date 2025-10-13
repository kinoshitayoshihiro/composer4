#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base instrument adapter for Stage3 v1.1 pipeline.

Provides common infrastructure for all instrument generators:
- Humanizer integration
- REMI tokenization
- Sidecar metadata (.meta.json)
- Consistent generate_one() API

Subclasses only need to implement:
  _build_pretty_midi(conditions, seed) -> pretty_midi.PrettyMIDI
"""
from __future__ import annotations
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Dependency flexibility (REMI/Humanizer location may vary)
try:
    from ml.tokenizer_remi import REMITokenizer
except Exception:
    try:
        from tokenizer_remi import REMITokenizer
    except Exception:
        REMITokenizer = None

try:
    from scripts.humanize_midi import humanize as humanize_pm
except Exception:
    def humanize_pm(pm, **_):  # no-op fallback
        return pm

try:
    import pretty_midi
except Exception as e:
    raise RuntimeError("pretty_midi required: pip install pretty_midi") from e


class BaseInstrumentAdapter:
    """
    Base class for instrument generators in Stage3 pipeline.

    Contract:
      - Subclass implements `_build_pretty_midi(conditions, seed) -> PrettyMIDI`
      - This class handles Humanizer/REMI/sidecar saving

    Returns from generate_one(..., save=True):
      {
        "pretty_midi": pm,
        "tokens": List[int],
        "midi_path": str (if save=True),
        "meta": dict (if save=True)
      }
    """
    part_name: str = "instrument"
    default_time_sig: str = "4/4"

    def __init__(
        self,
        *,
        out_dir: str = "output/gen",
        model_commit: str = "",
        tokenizer_hash: str = "REMI_1.1.0",
        remi_roles: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_commit = model_commit
        self.tokenizer_hash = tokenizer_hash
        self.remi_roles = remi_roles

        if REMITokenizer is not None:
            try:
                self.tokenizer = REMITokenizer.load_default()
            except Exception:
                self.tokenizer = REMITokenizer()
        else:
            self.tokenizer = None

    # ---- Abstract method for subclasses ----
    def _build_pretty_midi(self, conditions: Dict[str, Any], seed: int) -> "pretty_midi.PrettyMIDI":
        """
        Subclass MUST implement:
          Parse conditions dict and build a pretty_midi.PrettyMIDI object.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _build_pretty_midi()")

    # ---- Public API (common processing) ----
    def generate_one(
        self,
        *,
        conditions: Dict[str, Any],
        seed: int = 42,
        apply_humanizer: bool = True,
        humanizer_kwargs: Optional[Dict[str, Any]] = None,
        save: bool = True,
        file_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate single MIDI file with optional humanization and saving."""
        rng = random.Random(seed)  # Reserved for future probabilistic processing
        pm = self._build_pretty_midi(conditions, seed)

        if apply_humanizer:
            hk = humanizer_kwargs or {}
            pm = humanize_pm(
                pm,
                seed=seed,
                velocity_std=hk.get("velocity_std", 10.0),
                timing_jitter=hk.get("timing_jitter", 0.016),
                swing=hk.get("swing", 0.0),
                ar1=hk.get("ar1", 0.5),
            )

        tokens = []
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(pm, roles=self.remi_roles)
            except Exception:
                pass  # Tokenization optional

        out: Dict[str, Any] = {"pretty_midi": pm, "tokens": tokens}

        if save:
            stem = file_stem or self._default_stem(conditions, seed)
            midi_path = self.out_dir / f"{stem}.mid"
            midi_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            pm.write(str(midi_path))

            meta = self._make_sidecar_meta(midi_path, conditions, seed, token_count=len(tokens))
            sidecar = midi_path.with_suffix(".meta.json")
            sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            out["midi_path"] = str(midi_path)
            out["meta"] = meta

        return out

    # Batch generation utility
    def generate(
        self, n: int, *, base_conditions: Dict[str, Any], seed: int = 42, **kw
    ) -> List[Dict[str, Any]]:
        """Generate multiple MIDI files with incremental seeds."""
        results = []
        for i in range(n):
            c = dict(base_conditions)
            r = self.generate_one(conditions=c, seed=seed + i, **kw)
            results.append(r)
        return results

    # ---- Helpers ----
    def _default_stem(self, conditions: Dict[str, Any], seed: int) -> str:
        """Generate default filename stem from conditions."""
        tempo = conditions.get("tempo", 120)
        bars = conditions.get("length_bars", 16)
        style = conditions.get("style", "default")
        return f"{self.part_name}_{style}_{tempo}bpm_{bars}bars_seed{seed}"

    def _make_sidecar_meta(
        self, midi_path: Path, conditions: Dict[str, Any], seed: int, *, token_count: int
    ) -> Dict[str, Any]:
        """Create sidecar metadata for .meta.json file."""
        h = hashlib.sha1()
        h.update(str(midi_path).encode())
        h.update(json.dumps(conditions, sort_keys=True).encode())
        h.update(str(seed).encode())
        h.update(str(time.time()).encode())
        gen_id = h.hexdigest()[:16]

        return {
            "gen_id": gen_id,
            "part": self.part_name,
            "model_commit": self.model_commit,
            "tokenizer_hash": self.tokenizer_hash,
            "remi_version": "1.1.0",
            "created_at": time.time(),
            "conditions": conditions,
            "token_count": int(token_count),
            # Common keys for evaluator
            "tempo": conditions.get("tempo", 120),
            "time_sig": conditions.get("time_sig", self.default_time_sig),
            "length_bars": conditions.get("length_bars", 16),
        }
