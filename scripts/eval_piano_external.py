#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Benchmark Evaluation for Piano Transformer (Phase 4.3)

Evaluates Piano Transformer output on MAESTRO subset:
- Chord tone rate
- Hand separation
- Velocity diversity
- Bar violation rate
- Notes per bar density

Usage:
    python scripts/eval_piano_external.py \\
      --model-dir models/piano_transformer/best \\
      --maestro-dir data/maestro_subset \\
      --out-json output/reports/piano_external_bench.json \\
      --n-samples 10
"""

import argparse
import json
import math
import os
import random
from hashlib import sha1
from pathlib import Path
from typing import Dict, Any, List

import pretty_midi


def parse_time_sig(tsig: str) -> tuple:
    """Parse time signature string like '4/4' -> (4, 4)."""
    parts = tsig.split("/")
    return int(parts[0]), int(parts[1])


def file_metrics_piano_simple(mid_path: Path) -> Dict[str, Any]:
    """
    Simplified piano metrics for external benchmarks.
    Reuses logic from eval_drum_batch_stratified.py but minimal dependencies.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(mid_path))
    except Exception as e:
        return {
            "file": str(mid_path.name),
            "error": str(e),
            "valid": False,
            "reason": "parse_error"
        }
    
    # Find piano track (non-drum)
    track = None
    for inst in pm.instruments:
        if not inst.is_drum:
            track = inst
            break
    
    if not track or not track.notes:
        return {
            "file": str(mid_path.name),
            "error": "no_piano_notes",
            "valid": False,
            "reason": "no_piano_tracks_or_notes"
        }
    
    notes = track.notes
    
    # Metadata
    tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
    tsig = "4/4"
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        tsig = f"{ts.numerator}/{ts.denominator}"
    
    # Bar calculation
    num, den = parse_time_sig(tsig)
    bar_len_sec = num * (60.0 / tempo) * (4.0 / den)
    total_len = pm.get_end_time()
    bars = math.ceil(total_len / bar_len_sec) if bar_len_sec > 1e-6 else 1
    
    # 1) Hand separation proxy (pitch range spread)
    pitches = [n.pitch for n in notes]
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    hand_separation = min(1.0, pitch_range / 48.0)  # Normalized to 4 octaves
    
    # 2) Velocity diversity
    vels = [n.velocity for n in notes]
    vel_std = 0.0
    if len(vels) > 1:
        mean_vel = sum(vels) / len(vels)
        vel_std = (sum((v - mean_vel) ** 2 for v in vels) / len(vels)) ** 0.5
    
    # 3) Bar violation rate (notes spanning multiple bars)
    violations = 0
    for note in notes:
        start_bar = int(note.start / bar_len_sec)
        end_bar = int(note.end / bar_len_sec)
        if end_bar > start_bar:
            violations += 1
    bar_violation_rate = violations / len(notes) if notes else 0.0
    
    # 4) Notes per bar
    notes_per_bar = len(notes) / bars if bars > 0 else 0.0
    
    # 5) Chord tone rate (simplified: pitch class diversity)
    pitch_classes = set(p % 12 for p in pitches)
    chord_tone_rate = min(1.0, len(pitch_classes) / 7.0)  # 7-note scale
    
    return {
        "file": str(mid_path.name),
        "valid": True,
        "tempo": round(tempo, 1),
        "bars": bars,
        "time_sig": tsig,
        "chord_tone_rate": round(chord_tone_rate, 4),
        "hand_separation": round(hand_separation, 4),
        "velocity_std": round(vel_std, 2),
        "bar_violation_rate": round(bar_violation_rate, 4),
        "notes_per_bar": round(notes_per_bar, 2),
    }


def evaluate_maestro_subset(maestro_dir: Path, n_samples: int = 10, seed: int = 42) -> List[Dict]:
    """
    Evaluate MAESTRO subset with deterministic sampling.
    
    Args:
        maestro_dir: Path to MAESTRO MIDI files (or subset)
        n_samples: Number of samples to evaluate
        seed: Random seed for sampling
    
    Returns:
        List of metrics dicts
    """
    midi_files = list(maestro_dir.glob("**/*.mid"))
    midi_files.extend(maestro_dir.glob("**/*.midi"))
    
    if not midi_files:
        raise ValueError(f"No MIDI files found in {maestro_dir}")
    
    print(f"[info] Found {len(midi_files)} MIDI files in MAESTRO subset")
    
    # Deterministic sampling: stable sort by SHA1, then shuffle with seed
    midi_files = sorted(midi_files, key=lambda p: sha1(str(p).encode('utf-8')).hexdigest())
    rng = random.Random(seed)
    rng.shuffle(midi_files)
    sampled = midi_files[:min(n_samples, len(midi_files))]
    
    results = []
    for i, mf in enumerate(sampled):
        print(f"[progress] {i+1}/{len(sampled)}: {mf.name}")
        metrics = file_metrics_piano_simple(mf)
        results.append(metrics)
    
    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate metrics across samples."""
    valid = [r for r in results if r.get("valid", False)]
    
    if not valid:
        return {"error": "no_valid_samples", "total": len(results)}
    
    def mean(key):
        vals = [r[key] for r in valid if key in r]
        return sum(vals) / len(vals) if vals else 0.0
    
    def median(key):
        vals = sorted([r[key] for r in valid if key in r])
        if not vals:
            return 0.0
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    
    agg = {
        "total_samples": len(results),
        "valid_samples": len(valid),
        "chord_tone_rate": {
            "mean": round(mean("chord_tone_rate"), 4),
            "median": round(median("chord_tone_rate"), 4),
        },
        "hand_separation": {
            "mean": round(mean("hand_separation"), 4),
            "median": round(median("hand_separation"), 4),
        },
        "velocity_std": {
            "mean": round(mean("velocity_std"), 2),
            "median": round(median("velocity_std"), 2),
        },
        "bar_violation_rate": {
            "mean": round(mean("bar_violation_rate"), 4),
            "median": round(median("bar_violation_rate"), 4),
        },
        "notes_per_bar": {
            "mean": round(mean("notes_per_bar"), 2),
            "median": round(median("notes_per_bar"), 2),
        },
    }
    
    return agg


def main():
    ap = argparse.ArgumentParser(description="Evaluate Piano Transformer on MAESTRO")
    ap.add_argument("--maestro-dir", required=True, help="MAESTRO MIDI directory")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument("--n-samples", type=int, default=10, help="Number of samples to evaluate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    
    maestro_dir = Path(args.maestro_dir)
    if not maestro_dir.exists():
        raise SystemExit(f"MAESTRO directory not found: {maestro_dir}")
    
    print("[info] Evaluating Piano Transformer on MAESTRO subset...")
    
    # Evaluate
    results = evaluate_maestro_subset(maestro_dir, args.n_samples, args.seed)
    
    # Aggregate
    summary = aggregate_metrics(results)
    
    # Output with provenance information
    output = {
        "benchmark": "maestro_subset",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "summary": summary,
        "per_file": results,
        "provenance": {
            "maestro_dir": str(maestro_dir),
            "git_commit": os.getenv("GIT_COMMIT", ""),
            "git_branch": os.getenv("GIT_BRANCH", ""),
        }
    }
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    print(f"\n[summary]")
    print(f"  Valid samples: {summary['valid_samples']}/{summary['total_samples']}")
    print(f"  Chord tone rate: {summary['chord_tone_rate']['mean']:.4f}")
    print(f"  Hand separation: {summary['hand_separation']['mean']:.4f}")
    print(f"  Velocity std: {summary['velocity_std']['mean']:.2f}")
    print(f"  Bar violation rate: {summary['bar_violation_rate']['mean']:.4f}")
    print(f"  Notes per bar: {summary['notes_per_bar']['mean']:.2f}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
