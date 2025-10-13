#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StringsAdapter - BaseInstrumentAdapter継承のストリングス生成アダプタ

Styles:
- pad: パッドサステイン（長尺和音）
- pizz: ピチカート（短いスタッカート）

Density:
- low: 全音符持続 / 8分刻み
- mid: 2分音符×2 / 12分刻み
- high: 4分音符×4 / 16分刻み

Range: G2-E6 (MIDI 43-88)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
from pathlib import Path

try:
    import pretty_midi
except Exception as e:
    raise RuntimeError("pip install pretty_midi が必要です") from e

try:
    from adapters.base_instrument_adapter import BaseInstrumentAdapter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adapters.base_instrument_adapter import BaseInstrumentAdapter

# GM Program: 48 = String Ensemble 1
STRINGS_PROGRAM = 48
_FALLBACK_PROG = ["C", "G", "Am", "F"]


class StringsAdapter(BaseInstrumentAdapter):
    """Strings adapter with pad/pizz styles."""
    
    part_name = "strings"
    default_time_sig = "4/4"
    
    def __init__(self, *, out_dir: str = "output/gen_strings", program: int = STRINGS_PROGRAM, **kw):
        super().__init__(out_dir=out_dir, **kw)
        self.program = int(program)
        self.remi_roles = ["MELODY", "CHORD", "PAD"]
    
    def _build_pretty_midi(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        """Build PrettyMIDI from conditions (template engine)."""
        rng = random.Random(seed)
        
        tempo: float = float(conditions.get("tempo", 120))
        tsig: str = conditions.get("time_sig", self.default_time_sig)
        bars: int = int(conditions.get("length_bars", 16))
        style: str = conditions.get("style", "pad")  # "pad" | "pizz"
        density: str = conditions.get("density", "mid")  # "low" | "mid" | "high"
        
        # コード進行
        prog = self._attrs_to_progression(conditions) or _FALLBACK_PROG
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=self.program, is_drum=False, name="strings_ens")
        
        bar_len = self._bar_len_sec(tempo, tsig)
        chords = [self._name_to_pitches(nm) for nm in prog]
        
        for b in range(bars):
            t0 = b * bar_len
            chord = chords[b % len(chords)]
            
            if style == "pad":
                # low: 1bar持続 / mid: 半分に分割 / high: 4分刻み
                segs = 1 if density == "low" else 2 if density == "mid" else 4
                segL = bar_len / segs
                
                for s in range(segs):
                    st = t0 + s * segL
                    en = t0 + (s + 1) * segL - 0.02
                    
                    # 和音全音を重ねる（軽いジッタで不自然さ軽減）
                    for p in chord:
                        vel = 58 + rng.randint(-3, +4)
                        inst.notes.append(
                            pretty_midi.Note(
                                start=st,
                                end=en,
                                pitch=p,
                                velocity=max(1, min(127, vel)),
                            )
                        )
            else:
                # pizzicato: 8/12/16分相当で短く刻む
                steps = 8 if density == "low" else 12 if density == "mid" else 16
                step = bar_len / steps
                dur = min(0.18, step * 0.45)
                
                for i in range(steps):
                    p = chord[i % len(chord)]
                    st = t0 + i * step
                    en = st + dur
                    
                    # 拍頭アクセント+ジッタ
                    vel = 60 + (6 if i % (steps // 4 or 1) == 0 else 0)
                    vel += rng.randint(-4, +4)
                    
                    inst.notes.append(
                        pretty_midi.Note(
                            start=st,
                            end=en,
                            pitch=p,
                            velocity=max(1, min(127, vel)),
                        )
                    )
        
        pm.instruments.append(inst)
        
        # テンポ & 拍子
        pm._PrettyMIDI__tempo_changes = ([0.0], [tempo])
        num, den = self._parse_tsig(tsig)
        pm.time_signature_changes = [pretty_midi.TimeSignature(num, den, 0)]
        
        return pm
    
    # ========== Helpers ==========
    
    def _attrs_to_progression(self, conditions: Dict[str, Any]) -> List[str]:
        """Extract [chord:...] from attrs."""
        attrs = conditions.get("attrs", [])
        out = []
        for a in attrs:
            if a.startswith("[chord:") and a.endswith("]"):
                out.append(a[len("[chord:") : -1])
        return out
    
    @staticmethod
    def _name_to_pitches(name: str) -> List[int]:
        """Chord name → MIDI pitches (pad用に広げる、G2-E6レンジ)."""
        ROOTS = {
            "C": 0, "C#": 1, "Db": 1,
            "D": 2, "D#": 3, "Eb": 3,
            "E": 4,
            "F": 5, "F#": 6, "Gb": 6,
            "G": 7, "G#": 8, "Ab": 8,
            "A": 9, "A#": 10, "Bb": 10,
            "B": 11,
        }
        
        root = "".join([c for c in name if c.isalpha() or c in "#b"])
        if root not in ROOTS:
            return [60, 64, 67, 72]  # Cmaj7-ish fallback
        
        minor = ("m" in name and "maj" not in name)
        ext7 = ("7" in name)
        
        base = 60 + ROOTS[root]  # C4基準
        tri = [0, 3, 7] if minor else [0, 4, 7]
        pcs = [base + i for i in tri]
        
        if ext7:
            pcs.append(base + (10 if minor else 11))
        
        # pad用に広げる（下に根音、上にオクターブ追加）
        pcs = sorted(set(pcs + [min(88, max(43, pcs[0] - 12)), min(88, pcs[0] + 12)]))
        
        # レンジ制御: G2-E6 (43-88)
        pcs = [min(88, max(43, p)) for p in pcs]
        return pcs
    
    @staticmethod
    def _parse_tsig(s: str) -> Tuple[int, int]:
        try:
            a, b = s.split("/")
            return int(a), int(b)
        except Exception:
            return 4, 4
    
    @staticmethod
    def _bar_len_sec(bpm: float, tsig: str) -> float:
        a, b = StringsAdapter._parse_tsig(tsig)
        return a * (60.0 / float(bpm)) * (4.0 / b)


# ========== Standalone test ==========

def generate_strings(
    tempo: int = 120,
    time_sig: str = "4/4",
    length_bars: int = 16,
    style: str = "pad",
    density: str = "mid",
    chords: List[str] = None,
    seed: int = 42,
) -> pretty_midi.PrettyMIDI:
    """Standalone strings generation function for quick testing."""
    conditions = {
        "tempo": tempo,
        "time_sig": time_sig,
        "length_bars": length_bars,
        "style": style,
        "density": density,
    }
    if chords:
        conditions["attrs"] = [f"[chord:{ch}]" for ch in chords]
    
    adapter = StringsAdapter()
    return adapter._build_pretty_midi(conditions, seed)


if __name__ == "__main__":
    # Quick test
    pm = generate_strings(
        tempo=90,
        length_bars=8,
        style="pad",
        density="mid",
        chords=["C", "G", "Am", "F"],
        seed=42,
    )
    pm.write("test_strings_pad.mid")
    print("✅ Test MIDI written: test_strings_pad.mid")
