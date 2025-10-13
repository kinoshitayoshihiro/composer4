"""Thin adapter connecting legacy DrumGenerator to Stage3 v1.1 pipeline."""
from __future__ import annotations
from typing import Dict, Any, Optional
import random
from pathlib import Path

# Fallback imports（配置差を吸収）
try:
    from generator.drum_generator import DrumGenerator
except Exception:
    from drum_generator import DrumGenerator

try:
    from ml.tokenizer_remi import REMITokenizer
except Exception:
    try:
        from tokenizer_remi import REMITokenizer
    except Exception:
        REMITokenizer = None

try:
    from scripts.humanize_midi import humanize
except Exception:
    try:
        from humanize_midi import humanize
    except Exception:
        humanize = None


class DrumAdapter:
    """旧 DrumGenerator を Stage3 v1.1 に接続する薄いアダプタ。"""
    
    def __init__(self, patterns_dir: str = "data/drum_patterns"):
        self.patterns_dir = Path(patterns_dir)

    def generate_one(
        self,
        *,
        tempo: int = 120,
        time_sig: str = "4/4",
        length_bars: int = 64,
        style: str = "pop_straight",
        density: str = "mid",
        swing: float = 0.0,
        seed: int = 42,
        apply_humanizer: bool = True,
        humanizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate one drum pattern with the given parameters.
        
        Returns:
            Dict with keys: "pretty_midi", "tokens" (if REMITokenizer available)
        """
        rng = random.Random(seed)

        dg = DrumGenerator(
            part_name="drum",
            global_settings={
                "tempo": tempo,
                "time_signature": time_sig,
                "patterns_dir": str(self.patterns_dir),
            },
            main_cfg={
                "style": style,
                "density": density,
                "swing": swing,
            },
        )

        # Generate pattern - compose() メソッドのシグネチャを確認
        # section_data経由で呼び出す
        section_data = {
            "absolute_offset": 0,
            "length_in_measures": length_bars,
            "musical_intent": {
                "emotion": "default",
                "intensity": "medium"
            },
            "part_params": {
                "drums": {
                    "rhythm_key": style,
                    "density": density,
                    "swing": swing,
                }
            },
            "tempo": tempo,
            "time_signature": time_sig,
        }
        
        part = dg.compose(section_data=section_data)
        
        # Convert music21 Part to PrettyMIDI
        import pretty_midi
        from music21 import midi as m21midi
        
        # Create a Score and export to MIDI
        from music21 import stream
        score = stream.Score()
        score.append(part)
        
        # Export to PrettyMIDI via temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = tmp.name
        
        score.write('midi', tmp_path)
        pm = pretty_midi.PrettyMIDI(tmp_path)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)

        # Humanizer v1.1（AR(1)+BPM連動+拍LUT+スウィング）
        if apply_humanizer and humanize is not None:
            hk = humanizer_kwargs or {}
            pm = humanize(
                pm,
                seed=seed,
                velocity_std=hk.get("velocity_std", 12.0),
                timing_jitter=hk.get("timing_jitter", 0.018),
                swing=hk.get("swing", swing),
                ar1=hk.get("ar1", 0.6),
            )

        # REMI v1.1 tokenization
        tokens = []
        if REMITokenizer is not None:
            try:
                tok = REMITokenizer.load_default()
            except Exception:
                try:
                    tok = REMITokenizer()
                except Exception:
                    tok = None
            
            if tok is not None:
                try:
                    tokens = tok.encode(pm, roles=True)
                except Exception:
                    tokens = []
        
        return {"pretty_midi": pm, "tokens": tokens}
