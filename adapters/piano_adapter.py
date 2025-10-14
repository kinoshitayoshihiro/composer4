#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PianoAdapter - BaseInstrumentAdapter継承のピアノ生成アダプタ

Engines:
- template: コード進行ベースのブロックコード／アルペジオ（決定論、安全な土台）
- ml: 学習モデル（小型モデルやLoRA）で右手メロ＋左手伴奏
- transformer: Transformer推論（SDPA/Standard推奨、FlashはN≥2048のみ）

Styles:
- block: ブロックコード（全音符 or 2分音符）
- arpeggio: アルペジオ（8/12/16分音符）

Density:
- low: 音数少なめ（全音符、8分音符）
- mid: 標準（2分音符、12分音符）
- high: 音数多め（2分音符×2回、16分音符）
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import random
import pretty_midi
from pathlib import Path

try:
    from adapters.base_instrument_adapter import BaseInstrumentAdapter
except ImportError:
    # Fallback for different import paths
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adapters.base_instrument_adapter import BaseInstrumentAdapter

# フォールバック進行: I–V–vi–IV (C–G–Am–F)
_FALLBACK_PROG = ["C", "G", "Am", "F"]


class PianoAdapter(BaseInstrumentAdapter):
    """Piano adapter with template/ml/transformer engines."""
    
    part_name = "piano"
    default_time_sig = "4/4"
    
    def __init__(self, *, engine: str = "template", model_dir: Optional[str] = None, out_dir: str = "output/gen_piano", **kw):
        super().__init__(out_dir=out_dir, **kw)
        self.engine = engine
        self.model_dir = model_dir
        # Piano-specific REMI roles
        self.remi_roles = ["MELODY", "CHORD", "BASS"]
    
    def _build_pretty_midi(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        if self.engine == "template":
            return self._build_template(conditions, seed)
        elif self.engine == "transformer":
            return self._build_transformer(conditions, seed)
        elif self.engine == "ml":
            return self._build_ml(conditions, seed)
        else:
            raise ValueError(f"Unknown engine: {self.engine}")
    
    def _build_transformer(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        """Generate using Transformer model."""
        import torch
        from transformers import AutoModelForCausalLM
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from token_utils import load_remi_tokenizer, decode_ids_to_pm, sample_model, build_prefix_ids_from_conditions
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(self.model_dir).to(device)
        model.eval()
        
        tk = load_remi_tokenizer()
        prompt_ids = build_prefix_ids_from_conditions(tk, conditions)
        
        torch.manual_seed(seed)
        ids = sample_model(model, prompt_ids, max_new_tokens=256, temperature=1.0, top_p=0.9)
        pm = decode_ids_to_pm(tk, ids)
        return pm
    
    def _build_ml(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        """Generate using ML model (placeholder)."""
        # TODO: Implement piano_ml_generator integration
        raise NotImplementedError("ML engine not yet implemented for Piano")
    
    def _build_template(self, conditions: Dict[str, Any], seed: int) -> pretty_midi.PrettyMIDI:
        """Build PrettyMIDI from conditions using template engine."""
        rng = random.Random(seed)
        
        # Extract conditions
        tempo: float = float(conditions.get("tempo", 120))
        tsig: str = conditions.get("time_sig", self.default_time_sig)
        bars: int = int(conditions.get("length_bars", 16))
        style: str = conditions.get("style", "block")  # "block" | "arpeggio"
        density: str = conditions.get("density", "mid")  # "low" | "mid" | "high"
        
        # 1) コード進行 (attrs の [chord:...] 優先 → フォールバック)
        prog = self._attrs_to_progression(conditions) or _FALLBACK_PROG
        
        # 2) Template rendering
        pm = self._render_template(prog, tempo, tsig, bars, style, density, rng)
        
        # 3) ペダル (任意): 4分音符単位で薄く
        self._inject_pedal(pm, tempo, tsig, bars, strength=0.6)
        
        return pm
    
    # ========== Engines ==========
    
    def _render_template(
        self,
        prog: List[str],
        tempo: float,
        tsig: str,
        bars: int,
        style: str,
        density: str,
        rng: random.Random,
    ) -> pretty_midi.PrettyMIDI:
        """Template engine: deterministic block chords or arpeggios."""
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="piano_main")
        
        bar_len = self._bar_len_sec(tempo, tsig)
        chords = [self._name_to_pitches(nm) for nm in prog]
        
        for b in range(bars):
            chord = chords[b % len(chords)]
            t0 = b * bar_len
            
            if style == "arpeggio":
                # Arpeggio: 8/12/16 steps per bar
                steps = 8 if density == "low" else (12 if density == "mid" else 16)
                step = bar_len / steps
                
                for i in range(steps):
                    p = chord[i % len(chord)]
                    # Velocity variation
                    v = 62 + rng.randint(-4, 4) + (6 if i % 2 == 0 else -2)
                    v = max(1, min(127, v))
                    inst.notes.append(
                        pretty_midi.Note(
                            start=t0 + i * step,
                            end=t0 + i * step + step * 0.9,
                            pitch=p,
                            velocity=v,
                        )
                    )
                
                # Bass note (root - octave)
                root = min(chord) - 12
                inst.notes.append(
                    pretty_midi.Note(
                        start=t0,
                        end=t0 + step * 1.5,
                        pitch=root,
                        velocity=72 + rng.randint(-3, 3),
                    )
                )
            
            else:  # "block"
                # Block chords: whole note or half notes
                dur = bar_len if density == "low" else (bar_len / 2)
                reps = 1 if density == "low" else 2
                
                for h in range(reps):
                    # Right hand chord
                    for p in chord:
                        v = 64 + 5 * h + rng.randint(-3, 3)
                        inst.notes.append(
                            pretty_midi.Note(
                                start=t0 + h * dur,
                                end=t0 + (h + 1) * dur,
                                pitch=p,
                                velocity=max(1, min(127, v)),
                            )
                        )
                    
                    # Bass note (left hand)
                    root = min(chord) - 12
                    v = 72 + 8 * h + rng.randint(-4, 4)
                    inst.notes.append(
                        pretty_midi.Note(
                            start=t0 + h * dur,
                            end=t0 + h * dur + dur * 0.6,
                            pitch=root,
                            velocity=max(1, min(127, v)),
                        )
                    )
        
        pm.instruments.append(inst)
        
        # Tempo & time signature
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
                chord_name = a[len("[chord:") : -1]
                out.append(chord_name)
        return out
    
    @staticmethod
    def _name_to_pitches(name: str) -> List[int]:
        """Chord name → MIDI pitches (triad + optional 7th)."""
        ROOTS = {
            "C": 0, "C#": 1, "Db": 1,
            "D": 2, "D#": 3, "Eb": 3,
            "E": 4,
            "F": 5, "F#": 6, "Gb": 6,
            "G": 7, "G#": 8, "Ab": 8,
            "A": 9, "A#": 10, "Bb": 10,
            "B": 11,
        }
        
        # Parse root
        root_str = "".join([c for c in name if c.isalpha() or c in "#b"])
        if root_str not in ROOTS:
            return [60, 64, 67]  # C major triad fallback
        
        # Quality
        minor = ("m" in name and "maj" not in name)
        ext7 = ("7" in name)
        
        base = 60 + ROOTS[root_str]
        tri = [0, 3, 7] if minor else [0, 4, 7]
        pcs = [base + i for i in tri]
        
        if ext7:
            # Minor 7th or Major 7th
            pcs.append(base + (10 if minor else 11))
        
        return pcs
    
    @staticmethod
    def _parse_tsig(s: str) -> Tuple[int, int]:
        """Parse time signature string (e.g., '4/4' → (4, 4))."""
        try:
            a, b = s.split("/")
            return int(a), int(b)
        except Exception:
            return 4, 4
    
    @staticmethod
    def _bar_len_sec(bpm: float, tsig: str) -> float:
        """Bar length in seconds."""
        a, b = PianoAdapter._parse_tsig(tsig)
        return a * (60.0 / float(bpm)) * (4.0 / b)
    
    def _inject_pedal(
        self,
        pm: pretty_midi.PrettyMIDI,
        tempo: float,
        tsig: str,
        bars: int,
        strength: float = 0.6,
    ) -> None:
        """Inject CC64 (Sustain) pedal events.
        
        簡易にバー頭ON → 小節末OFF
        """
        bar_len = self._bar_len_sec(tempo, tsig)
        track = pretty_midi.Instrument(program=0, is_drum=False, name="piano_pedal_cc")
        
        for b in range(bars):
            # Pedal ON at bar start
            val_on = int(64 + strength * 63)
            track.control_changes.append(
                pretty_midi.ControlChange(number=64, value=val_on, time=b * bar_len)
            )
            # Pedal OFF at bar end
            track.control_changes.append(
                pretty_midi.ControlChange(number=64, value=0, time=(b + 1) * bar_len - 0.02)
            )
        
        pm.instruments.append(track)


# ========== Standalone test ==========

def generate_piano(
    tempo: int = 120,
    time_sig: str = "4/4",
    length_bars: int = 16,
    style: str = "block",
    density: str = "mid",
    engine: str = "template",
    chords: Optional[List[str]] = None,
    seed: int = 42,
) -> pretty_midi.PrettyMIDI:
    """Standalone piano generation function for quick testing."""
    conditions = {
        "tempo": tempo,
        "time_sig": time_sig,
        "length_bars": length_bars,
        "style": style,
        "density": density,
        "engine": engine,
    }
    if chords:
        conditions["attrs"] = [f"[chord:{ch}]" for ch in chords]
    
    adapter = PianoAdapter()
    return adapter._build_pretty_midi(conditions, seed)


if __name__ == "__main__":
    # Quick test
    pm = generate_piano(
        tempo=120,
        length_bars=8,
        style="block",
        density="mid",
        chords=["C", "G", "Am", "F"],
        seed=42,
    )
    pm.write("test_piano_template.mid")
    print("✅ Test MIDI written: test_piano_template.mid")
