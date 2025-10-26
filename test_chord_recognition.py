"""
Chord Recognition System Test Script

Tests the newly implemented librosa-based chord recognition system.
"""
from analysis.stem_harmony import (
    estimate_chords_per_stem,
    aggregate_stem_chords,
    make_beat_grid,
)
import os

def test_chord_recognition():
    """Test chord recognition on a sample WAV file."""
    
    # Test data path
    test_wav = "data/suno_ai/suno_themesong/song_001/stemswav_001/bass.wav"
    
    if not os.path.exists(test_wav):
        print(f"[ERROR] Test file not found: {test_wav}")
        print("Please provide a valid WAV file path for testing.")
        return
    
    print("="*60)
    print("Chord Recognition System Test")
    print("="*60)
    
    # Step 1: Create beat grid
    print("\n[Step 1] Creating beat grid (BPM=120, 4/4)...")
    stems_dummy = {
        "bass": test_wav,
        "guitar": test_wav,  # Using same file for demo
    }
    beat_grid = make_beat_grid(stems_dummy, default_bpm=120.0, time_sig=(4, 4))
    print(f"  BPM: {beat_grid['bpm']}")
    print(f"  Time signature: {beat_grid['time_sig']}")
    print(f"  Total beats: {len(beat_grid['beats'])}")
    print(f"  Total bars: {len(beat_grid['bars'])}")
    
    # Step 2: Estimate chords from bass stem
    print("\n[Step 2] Estimating chords from bass stem...")
    print(f"  Input: {test_wav}")
    print(f"  Key hint: C:maj")
    
    votes_bass = estimate_chords_per_stem(
        test_wav,
        beat_grid,
        role="bass",
        key_hint="C:maj",
        top_n=2,
    )
    
    print(f"  Result: {len(votes_bass)} beat positions analyzed")
    
    # Show first 8 beats
    print("\n  First 8 beats:")
    for i, ((bar, beat), candidates) in enumerate(list(votes_bass.items())[:8]):
        top_chord = candidates[0] if candidates else {"chord": "N", "score": 0.0}
        print(f"    Bar {bar}, Beat {beat}: {top_chord['chord']} (score={top_chord['score']:.3f})")
    
    # Step 3: Aggregate with voting system
    print("\n[Step 3] Aggregating with voting system...")
    
    stem_votes = {
        "bass": votes_bass,
    }
    
    activity = {
        "bass": [(i, 0.8) for i in range(len(beat_grid['bars']))],  # Dummy activity
    }
    
    cfg = {
        "weights": {"bass": 1.0},  # Only bass for this test
        "confidence_threshold": 0.3,
        "min_confidence_warn": 0.4,
    }
    
    audio_chordmap = aggregate_stem_chords(
        stem_votes,
        activity,
        key_hint="C:maj",
        sections=[],
        cfg=cfg,
    )
    
    print(f"  Key: {audio_chordmap['key']}")
    print(f"  Key confidence: {audio_chordmap['confidence_key']:.3f}")
    print(f"  Total chord items: {len(audio_chordmap['items'])}")
    print(f"  Low confidence warnings: {len(audio_chordmap.get('low_confidence_warnings', []))}")
    
    # Show first 8 chord items
    print("\n  First 8 chord items:")
    for item in audio_chordmap['items'][:8]:
        print(f"    Bar {item['bar']}, Beat {item['beat']}: {item['chord']} (conf={item['confidence']:.3f})")
    
    # Summary statistics
    chords_used = {}
    for item in audio_chordmap['items']:
        c = item['chord']
        chords_used[c] = chords_used.get(c, 0) + 1
    
    print("\n[Summary] Chord distribution:")
    for chord, count in sorted(chords_used.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(audio_chordmap['items'])
        print(f"  {chord}: {count} times ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_chord_recognition()
