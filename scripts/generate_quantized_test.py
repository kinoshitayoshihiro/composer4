#!/usr/bin/env python3
"""Generate quantized MIDI for humanizer testing."""

import pretty_midi

# Create a quantized drum pattern (uniform velocity, perfect timing)
midi = pretty_midi.PrettyMIDI(initial_tempo=120)
drums = pretty_midi.Instrument(program=0, is_drum=True)

# 4 bars of simple rock beat: kick-snare-kick-snare
# All notes at velocity=100 (completely uniform)
pattern = [
    # Bar 1
    (0.0, 36, 100),   # Kick
    (0.5, 38, 100),   # Snare
    (1.0, 36, 100),   # Kick
    (1.5, 38, 100),   # Snare
    # Bar 2
    (2.0, 36, 100),
    (2.5, 38, 100),
    (3.0, 36, 100),
    (3.5, 38, 100),
    # Bar 3
    (4.0, 36, 100),
    (4.5, 38, 100),
    (5.0, 36, 100),
    (5.5, 38, 100),
    # Bar 4
    (6.0, 36, 100),
    (6.5, 38, 100),
    (7.0, 36, 100),
    (7.5, 38, 100),
]

for start, pitch, velocity in pattern:
    note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=start,
        end=start + 0.1,
    )
    drums.notes.append(note)

# Add drums to MIDI
midi.instruments.append(drums)

midi.write("/tmp/quantized_test.mid")
print("✅ Created /tmp/quantized_test.mid (16 notes, velocity_std=0.0)")
