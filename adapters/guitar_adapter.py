#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GuitarAdapter - BaseInstrumentAdapter継承のギター生成アダプタ

Styles:
- strum: ストラム奏法（D/U交互、14msオフセット）
- arpeggio: アルペジオ（順次分散和音）

Density:
- low: 8分刻み
- mid: 12分刻み（8分+裏）
- high: 16分刻み

Range: E2-E5 (MIDI 40-76)
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

# GM Program: 24 = Acoustic Guitar (nylon), 25 = Steel
GUITAR_PROGRAM = 24
_FALLBACK_PROG = ["C", "G", "Am", "F"]


class GuitarAdapter(BaseInstrumentAdapter):
    """Guitar adapter with strum/arpeggio styles."""
    
    part_name = "guitar"
    default_time_sig = "4/4"
    
    def __init__(self, *, out_dir: str = "output/gen_guitar", program: int = GUITAR_PROGRAM, **kw):
        super().__init__(out_dir=out_dir, **kw)
        self.program = int(program)
        self.remi_roles = ["MELODY", "CHORD"]
    
    def _build_pretty_midi(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        """Build PrettyMIDI from conditions (template engine)."""
        rng = random.Random(seed)
        
        tempo: float = float(conditions.get("tempo", 120))
        tsig: str = conditions.get("time_sig", self.default_time_sig)
        bars: int = int(conditions.get("length_bars", 16))
        style: str = conditions.get("style", "strum")  # "strum" | "arpeggio"
        density: str = conditions.get("density", "mid")  # "low" | "mid" | "high"
        
        # コード進行
        prog = self._attrs_to_progression(conditions) or _FALLBACK_PROG
        
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=self.program, is_drum=False, name="guitar_main")
        
        bar_len = self._bar_len_sec(tempo, tsig)
        chords = [self._name_to_pitches(nm) for nm in prog]
        
        # ビート単位設定
        num_beats = self._parse_tsig(tsig)[0]
        beat = bar_len / num_beats
        
        # ストラム設定（10ms窓を越える14msオフセット）
        STRUM_STEP = 0.014  # 14ms
        
        if style == "strum":
            # low: 8分 / mid: 12分 / high: 16分
            grid = 2 if density == "low" else 3 if density == "mid" else 4
        else:
            # arpeggio: 同様に密度調整
            grid = 3 if density == "low" else 4 if density == "mid" else 6
        
        for b in range(bars):
            t0 = b * bar_len
            chord = chords[b % len(chords)]
            
            if style == "strum":
                # 各ビートでD(順)/U(逆)を交互に掃く
                for k in range(num_beats):
                    bt = t0 + k * beat
                    direction_down = (k % 2 == 0)
                    seq = chord if direction_down else list(reversed(chord))
                    
                    # grid回叩く（同和音を刻む）
                    for g in range(grid):
                        base = bt + g * (beat / grid)
                        for j, p in enumerate(seq):
                            st = base + j * STRUM_STEP
                            # ノート長: ビート単位の90%程度
                            dur = min(beat / grid - 0.005, 0.18 + 0.04 * rng.random())
                            en = st + dur
                            
                            # ベロシティ: 拍頭アクセント+ジッタ
                            vel = 62 + (6 if g == 0 and direction_down and j == 0 else 0)
                            vel += rng.randint(-5, +5)
                            
                            inst.notes.append(
                                pretty_midi.Note(
                                    start=st,
                                    end=en,
                                    pitch=p,
                                    velocity=max(1, min(127, vel)),
                                )
                            )
            else:
                # アルペジオ: 1ビートあたりgridノート
                step = beat / grid
                total = num_beats * grid
                
                for i in range(total):
                    p = chord[i % len(chord)]
                    st = t0 + i * step
                    en = min(t0 + (i + 1) * step - 0.006, st + step * 0.9)
                    
                    # 拍頭アクセント+ジッタ
                    vel = 64 + (4 if i % grid == 0 else 0) + rng.randint(-4, +4)
                    
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
        """Chord name → MIDI pitches (E4基準、E2-E5レンジ)."""
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
            return [64, 67, 71]  # E4/G4/B4 fallback
        
        minor = ("m" in name and "maj" not in name)
        ext7 = ("7" in name)
        
        base = 64 + ROOTS[root]  # E4基準
        tri = [0, 3, 7] if minor else [0, 4, 7]
        pcs = [base + i for i in tri]
        
        if ext7:
            pcs.append(base + (10 if minor else 11))
        
        # レンジ制御: E2-E5 (40-76)
        pcs = [min(76, max(40, p)) for p in pcs]
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
        a, b = GuitarAdapter._parse_tsig(tsig)
        return a * (60.0 / float(bpm)) * (4.0 / b)


# ========== Standalone test ==========

def generate_guitar(
    tempo: int = 120,
    time_sig: str = "4/4",
    length_bars: int = 16,
    style: str = "strum",
    density: str = "mid",
    chords: List[str] = None,
    seed: int = 42,
) -> pretty_midi.PrettyMIDI:
    """Standalone guitar generation function for quick testing."""
    conditions = {
        "tempo": tempo,
        "time_sig": time_sig,
        "length_bars": length_bars,
        "style": style,
        "density": density,
    }
    if chords:
        conditions["attrs"] = [f"[chord:{ch}]" for ch in chords]
    
    adapter = GuitarAdapter()
    return adapter._build_pretty_midi(conditions, seed)


if __name__ == "__main__":
    # Quick test
    pm = generate_guitar(
        tempo=120,
        length_bars=8,
        style="strum",
        density="mid",
        chords=["C", "G", "Am", "F"],
        seed=42,
    )
    pm.write("test_guitar_strum.mid")
    print("✅ Test MIDI written: test_guitar_strum.mid")
