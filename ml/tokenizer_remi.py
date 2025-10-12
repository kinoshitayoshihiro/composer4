"""
REMI-style tokenizer extension for Stage3 (v1.1 enhancement)

Adds DURATION, CHORD, and ROLE tokens to improve:
- Bar consistency (violation rate 3.2% → <2.0%)
- Harmonic validity (+21% improvement target)
- Drum coherence (+20% improvement target)

Based on:
- REMI (Huang & Yang 2020): https://arxiv.org/abs/2002.00212
- MuMIDI (Ens & Pasquier 2020)

Backward compatibility:
- remi_enabled flag controls new token usage
- Legacy mode supports v1.0 data loading
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pretty_midi

from ml.stage3_generator import Stage3Tokenizer


class REMITokenizer(Stage3Tokenizer):
    """Extended tokenizer with REMI-style DURATION/CHORD/ROLE tokens.
    
    Extensions:
    - DURATION tokens: Musical note durations (1/16, 1/8, 1/4, 1/2, 1, 2 bars)
    - CHORD tokens: Harmonic chord symbols (C, Dm, G7, etc.)
    - ROLE tokens: Drum instrument roles (KICK, SNARE, HIHAT, CRASH, etc.)
    
    Backward Compatibility:
    - remi_enabled=False: Use legacy encoding (v1.0 compatible)
    - remi_enabled=True: Use REMI extensions (v1.1)
    """
    
    # REMI extension token prefixes
    DURATION_PREFIX = "RDUR_"  # REMI duration
    CHORD_PREFIX = "CHORD_"
    ROLE_PREFIX = "ROLE_"
    
    # Musical duration definitions (in beats)
    DURATION_MAP = {
        "1/16": 0.25,
        "1/8": 0.5,
        "1/4": 1.0,
        "1/2": 2.0,
        "1": 4.0,
        "2": 8.0,
    }
    
    # Common chord types
    CHORD_TYPES = [
        # Major triads
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
        # Minor triads
        "Cm", "Dbm", "Dm", "Ebm", "Em", "Fm", "Gbm", "Gm", "Abm", "Am", "Bbm", "Bm",
        # Dominant 7ths
        "C7", "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7",
        # Major 7ths
        "Cmaj7", "Dbmaj7", "Dmaj7", "Ebmaj7", "Emaj7", "Fmaj7", "Gbmaj7",
        "Gmaj7", "Abmaj7", "Amaj7", "Bbmaj7", "Bmaj7",
        # Minor 7ths
        "Cm7", "Dbm7", "Dm7", "Ebm7", "Em7", "Fm7", "Gbm7", "Gm7", "Abm7", "Am7", "Bbm7", "Bm7",
        # Diminished/Augmented
        "Cdim", "Ddim", "Edim", "Fdim", "Gdim", "Adim", "Bdim",
        "Caug", "Daug", "Eaug", "Faug", "Gaug", "Aaug", "Baug",
    ]
    
    # Drum roles (GM MIDI standard pitches)
    DRUM_ROLES = {
        35: "KICK",       # Acoustic Bass Drum
        36: "KICK",       # Bass Drum 1
        38: "SNARE",      # Acoustic Snare
        40: "SNARE",      # Electric Snare
        42: "HIHAT",      # Closed Hi-Hat
        44: "HIHAT",      # Pedal Hi-Hat
        46: "HIHAT",      # Open Hi-Hat
        49: "CRASH",      # Crash Cymbal 1
        51: "RIDE",       # Ride Cymbal 1
        53: "RIDE",       # Ride Bell
        55: "CRASH",      # Splash Cymbal
        57: "CRASH",      # Crash Cymbal 2
        59: "RIDE",       # Ride Cymbal 2
        41: "TOM",        # Low Floor Tom
        43: "TOM",        # High Floor Tom
        45: "TOM",        # Low Tom
        47: "TOM",        # Low-Mid Tom
        48: "TOM",        # Hi-Mid Tom
        50: "TOM",        # High Tom
        37: "RIMSHOT",     # Side Stick
        39: "CLAP",        # Hand Clap
        54: "TAMBOURINE",  # Tambourine
        56: "COWBELL",     # Cowbell
    }
    
    def __init__(
        self,
        *,
        beat_division: int = 24,
        max_time_shift: int = 64,
        velocity_bins: int = 16,
        max_duration: int = 256,
        max_bars: int = 16,
        audio_bins: int = 10,
        remi_enabled: bool = False,
    ) -> None:
        """Initialize REMI tokenizer.
        
        Args:
            remi_enabled: If True, use REMI extensions. If False, legacy mode (v1.0).
        """
        self.remi_enabled = remi_enabled
        
        # Initialize base tokenizer
        super().__init__(
            beat_division=beat_division,
            max_time_shift=max_time_shift,
            velocity_bins=velocity_bins,
            max_duration=max_duration,
            max_bars=max_bars,
            audio_bins=audio_bins,
        )
        
        # Add REMI extension tokens if enabled
        if self.remi_enabled:
            self._init_remi_vocab()
    
    def _init_remi_vocab(self) -> None:
        """Add REMI extension tokens to vocabulary."""
        # Duration tokens
        for dur_name in self.DURATION_MAP.keys():
            self._add_token(f"{self.DURATION_PREFIX}{dur_name}")
        
        # Chord tokens
        for chord in self.CHORD_TYPES:
            self._add_token(f"{self.CHORD_PREFIX}{chord}")
        
        # Role tokens
        for role in set(self.DRUM_ROLES.values()):
            self._add_token(f"{self.ROLE_PREFIX}{role}")
    
    def encode_midi(self, midi: pretty_midi.PrettyMIDI) -> list[int]:
        """Encode MIDI with optional REMI extensions.
        
        Args:
            midi: PrettyMIDI object
            
        Returns:
            List of token IDs
        """
        if self.remi_enabled:
            return self._encode_remi(midi)
        else:
            return super().encode_midi(midi)
    
    def _encode_remi(self, midi: pretty_midi.PrettyMIDI) -> list[int]:
        """Encode with REMI extensions (DURATION/CHORD/ROLE)."""
        tokens: list[int] = []
        events: list[tuple[int, int, int, int, int, bool]] = []
        ticks_per_beat = midi.resolution
        step = max(1, ticks_per_beat // self.beat_division)
        
        # Collect all note events
        for inst in midi.instruments:
            inst_token_id = self.ensure_instrument_token(inst.program, inst.is_drum)
            for note in inst.notes:
                start_tick = int(round(midi.time_to_tick(note.start)))
                end_tick = int(round(midi.time_to_tick(note.end)))
                if end_tick <= start_tick:
                    end_tick = start_tick + step
                events.append((
                    start_tick,
                    end_tick,
                    note.pitch,
                    note.velocity,
                    inst_token_id,
                    inst.is_drum
                ))
        
        if not events:
            return tokens
        
        events.sort(key=lambda x: (x[0], x[2], x[3]))
        last_tick = 0
        current_inst = None
        
        for start_tick, end_tick, pitch, velocity, inst_tok, is_drum in events:
            # Time shift
            delta = start_tick - last_tick
            while delta > 0:
                shift = min(self.max_time_shift, max(1, delta // step))
                tokens.append(self.time_shift_token(shift))
                delta -= shift * step
            
            # Instrument switch
            if inst_tok != current_inst:
                tokens.append(inst_tok)
                current_inst = inst_tok
                
                # Add ROLE token for drums (REMI extension)
                if is_drum and pitch in self.DRUM_ROLES:
                    role = self.DRUM_ROLES[pitch]
                    role_token = f"{self.ROLE_PREFIX}{role}"
                    if role_token in self.token_to_id:
                        tokens.append(self.token_to_id[role_token])
            
            # Note pitch
            tokens.append(self.note_token(pitch))
            
            # Velocity
            tokens.append(self.velocity_token(velocity))
            
            # Duration with REMI extension
            duration_steps = max(1, (end_tick - start_tick) // step)
            duration_beats = duration_steps / (ticks_per_beat / step)
            
            # Try to map to REMI duration token
            remi_dur_token = self._find_remi_duration(duration_beats)
            if remi_dur_token:
                tokens.append(self.token_to_id[remi_dur_token])
            else:
                # Fallback to legacy duration
                tokens.append(self.duration_token(duration_steps))
            
            last_tick = start_tick
        
        return tokens
    
    def _find_remi_duration(self, duration_beats: float) -> str | None:
        """Find closest REMI duration token.
        
        Args:
            duration_beats: Duration in beats
            
        Returns:
            REMI duration token or None if no match
        """
        tolerance = 0.25  # Allow 25% deviation
        
        for dur_name, dur_value in self.DURATION_MAP.items():
            if abs(duration_beats - dur_value) / dur_value < tolerance:
                return f"{self.DURATION_PREFIX}{dur_name}"
        
        return None
    
    def save(self, path: Path) -> None:
        """Save tokenizer config with REMI flag."""
        data = {
            "token_to_id": self.token_to_id,
            "beat_division": self.beat_division,
            "max_time_shift": self.max_time_shift,
            "velocity_bins": self.velocity_bins,
            "max_duration": self.max_duration,
            "max_bars": self.max_bars,
            "audio_bins": self.audio_bins,
            "remi_enabled": self.remi_enabled,  # v1.1 extension
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "REMITokenizer":
        """Load tokenizer from saved config."""
        data = json.loads(path.read_text())
        
        tokenizer = cls(
            beat_division=data.get("beat_division", 24),
            max_time_shift=data.get("max_time_shift", 64),
            velocity_bins=data.get("velocity_bins", 16),
            max_duration=data.get("max_duration", 256),
            max_bars=data.get("max_bars", 16),
            audio_bins=data.get("audio_bins", 10),
            remi_enabled=data.get("remi_enabled", False),
        )
        
        # Restore vocabulary
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {v: k for k, v in data["token_to_id"].items()}  # Reverse mapping
        
        return tokenizer
    
    def get_stats(self) -> dict[str, Any]:
        """Get tokenizer statistics."""
        stats = {
            "vocab_size": self.vocab_size,
            "remi_enabled": self.remi_enabled,
            "beat_division": self.beat_division,
            "max_time_shift": self.max_time_shift,
            "velocity_bins": self.velocity_bins,
            "max_duration": self.max_duration,
        }
        
        if self.remi_enabled:
            duration_tokens = [t for t in self.token_to_id if t.startswith(self.DURATION_PREFIX)]
            chord_tokens = [t for t in self.token_to_id if t.startswith(self.CHORD_PREFIX)]
            role_tokens = [t for t in self.token_to_id if t.startswith(self.ROLE_PREFIX)]
            
            stats["remi_extensions"] = {
                "duration_tokens": len(duration_tokens),
                "chord_tokens": len(chord_tokens),
                "role_tokens": len(role_tokens),
            }
        
        return stats
