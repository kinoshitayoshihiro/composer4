#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bass Generator Evaluation Script (Phase 4.6)

Evaluates Bass Generator output:
- root_or_chord_tone_rate: ルート/和声音ヒット率
- leap_rate: 跳躍の割合
- max_leap_semitones: 最大跳躍幅
- grid_off_std_ms: タイミング安定度
- notes_per_bar: 音符密度
- velocity_std: ベロシティ分散

Usage:
    python scripts/eval_bass.py \\
      --input output/bass/*.mid \\
      --out-json output/reports/bass_eval.json \\
      --out-csv output/reports/bass_eval.csv
"""

import argparse
import csv
import json
import math
import statistics
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pretty_midi

# Schema version for output JSON
SCHEMA_VERSION = "1.1"

# Bass range constants (from eval_drum_batch_stratified.py)
BASS_RANGE_MIN = 28  # E1
BASS_RANGE_MAX = 55  # G3

# Threshold definitions (aligned with config/quality_gates.yaml)
THRESHOLDS = {
    "root_or_chord_tone_rate": {"min": 0.70, "direction": "higher_is_better"},
    "leap_rate": {"max": 0.20, "direction": "lower_is_better"},
    "max_leap_semitones": {"max": 12, "direction": "lower_is_better"},
    "grid_off_std_ms": {"max": 18, "direction": "lower_is_better"},
    "notes_per_bar": {"min": 4.0, "max": 12.0, "direction": "within_range"},
    "velocity_std": {"min": 10.0, "max": 22.0, "direction": "within_range"},
}

# Chord root mapping for root/chord tone detection
CHORD_ROOTS = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

# Scale intervals (major/minor)
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _fileset_hash(paths: List[Path]) -> str:
    """Calculate SHA1 hash of the sampled file set."""
    rels = [str(p) for p in paths]
    blob = "\n".join(sorted(rels)).encode("utf-8")
    return sha1(blob).hexdigest()


def _flag_metric(name: str, val: Optional[float]) -> Optional[str]:
    """Check if metric value violates threshold."""
    t = THRESHOLDS.get(name)
    if t is None or val is None:
        return None
    
    direction = t.get("direction")
    
    if direction == "higher_is_better" and val < t.get("min", float('-inf')):
        return f"{name}:low"
    
    if direction == "lower_is_better" and val > t.get("max", float('inf')):
        return f"{name}:high"
    
    if direction == "within_range":
        if val < t.get("min", float('-inf')):
            return f"{name}:low"
        if val > t.get("max", float('inf')):
            return f"{name}:high"
    
    return None


def parse_time_sig(tsig: str) -> Tuple[int, int]:
    """Parse time signature string like '4/4' -> (4, 4)."""
    parts = tsig.split("/")
    return int(parts[0]), int(parts[1])


def bar_len_sec(bpm: float, tsig: str) -> float:
    """Calculate bar length in seconds."""
    num, den = parse_time_sig(tsig)
    return num * (60.0 / float(bpm)) * (4.0 / den)


def is_chord_tone(pitch: int, chord_root: str, mode: str = "major") -> bool:
    """
    Check if pitch is a chord tone relative to given root.
    
    Args:
        pitch: MIDI pitch number
        chord_root: Root note name (e.g., "C", "D#")
        mode: "major" or "minor"
    
    Returns:
        True if pitch is a scale tone (chord tone approximation)
    """
    root_pc = CHORD_ROOTS.get(chord_root, 0)
    pitch_pc = pitch % 12
    
    intervals = MAJOR_INTERVALS if mode == "major" else MINOR_INTERVALS
    target_pcs = [(root_pc + i) % 12 for i in intervals]
    
    return pitch_pc in target_pcs


def file_metrics_bass(mid_path: Path, chord_root: str = "C", mode: str = "major") -> Dict[str, Any]:
    """
    Calculate bass metrics for a single MIDI file.
    
    Args:
        mid_path: Path to MIDI file
        chord_root: Root note for chord tone detection (default: "C")
        mode: "major" or "minor" (default: "major")
    
    Returns:
        Dictionary with metrics and metadata
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
    
    # Find bass track (non-drum, within bass range)
    bass_track = None
    for inst in pm.instruments:
        if not inst.is_drum and inst.notes:
            # Check if most notes are in bass range
            bass_notes = [n for n in inst.notes if BASS_RANGE_MIN <= n.pitch <= BASS_RANGE_MAX]
            if len(bass_notes) >= len(inst.notes) * 0.5:
                bass_track = inst
                break
    
    if not bass_track or not bass_track.notes:
        return {
            "file": str(mid_path.name),
            "error": "no_bass_notes",
            "valid": False,
            "reason": "no_bass_tracks_or_notes"
        }
    
    notes = sorted(bass_track.notes, key=lambda n: n.start)
    
    # Metadata
    tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
    tsig = "4/4"
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        tsig = f"{ts.numerator}/{ts.denominator}"
    
    # Bar calculation
    num, den = parse_time_sig(tsig)
    bar_length = bar_len_sec(tempo, tsig)
    total_len = pm.get_end_time()
    bars = math.ceil(total_len / bar_length) if bar_length > 1e-6 else 1
    
    # 1) root_or_chord_tone_rate
    chord_tone_count = sum(1 for n in notes if is_chord_tone(n.pitch, chord_root, mode))
    root_or_chord_tone_rate = chord_tone_count / len(notes) if notes else 0.0
    
    # 2) leap_rate & max_leap_semitones
    leaps = []
    for i in range(1, len(notes)):
        interval = abs(notes[i].pitch - notes[i-1].pitch)
        leaps.append(interval)
    
    leap_threshold = 3  # Major 3rd or larger
    leap_count = sum(1 for leap in leaps if leap >= leap_threshold)
    leap_rate = leap_count / len(leaps) if leaps else 0.0
    max_leap_semitones = max(leaps) if leaps else 0
    
    # 3) grid_off_std_ms (timing stability)
    # Calculate quantization grid based on 16th notes
    sixteenth_duration = bar_length / 16.0
    grid_times = []
    t = 0.0
    while t < total_len:
        grid_times.append(t)
        t += sixteenth_duration
    
    grid_offs = []
    for n in notes:
        # Find nearest grid point
        nearest_dist = min(abs(n.start - g) for g in grid_times)
        grid_offs.append(nearest_dist * 1000.0)  # Convert to ms
    
    grid_off_std_ms = statistics.stdev(grid_offs) if len(grid_offs) > 1 else 0.0
    
    # 4) notes_per_bar
    notes_per_bar = len(notes) / bars if bars > 0 else 0.0
    
    # 5) velocity_std
    vels = [n.velocity for n in notes]
    velocity_std = statistics.stdev(vels) if len(vels) > 1 else 0.0
    
    # Assemble result
    metrics = {
        "root_or_chord_tone_rate": round(root_or_chord_tone_rate, 3),
        "leap_rate": round(leap_rate, 3),
        "max_leap_semitones": max_leap_semitones,
        "grid_off_std_ms": round(grid_off_std_ms, 2),
        "notes_per_bar": round(notes_per_bar, 2),
        "velocity_std": round(velocity_std, 2),
    }
    
    return {
        "file": str(mid_path.name),
        "valid": True,
        "tempo": round(tempo, 1),
        "time_signature": tsig,
        "bars": bars,
        "note_count": len(notes),
        "chord_root": chord_root,
        "mode": mode,
        "metrics": metrics,
    }


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across multiple files."""
    valid = [r for r in results if r.get("valid", False)]
    
    if not valid:
        return {
            "n_files": len(results),
            "n_valid": 0,
            "n_invalid": len(results),
            "error": "no_valid_files"
        }
    
    # Collect all metric values
    metric_names = list(THRESHOLDS.keys())
    metric_values = {name: [] for name in metric_names}
    
    for r in valid:
        metrics = r.get("metrics", {})
        for name in metric_names:
            if name in metrics and metrics[name] is not None:
                metric_values[name].append(metrics[name])
    
    # Calculate statistics
    summary = {}
    for name in metric_names:
        vals = metric_values[name]
        if vals:
            summary[name] = {
                "mean": round(statistics.mean(vals), 3),
                "median": round(statistics.median(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
        else:
            summary[name] = None
    
    return {
        "n_files": len(results),
        "n_valid": len(valid),
        "n_invalid": len(results) - len(valid),
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bass Generator output")
    parser.add_argument("--input", required=True, help="Input MIDI files (glob pattern)")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-csv", help="Output CSV path (optional)")
    parser.add_argument("--chord-root", default="C", help="Chord root for analysis (default: C)")
    parser.add_argument("--mode", default="major", choices=["major", "minor"], help="Scale mode")
    
    args = parser.parse_args()
    
    # Collect input files
    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.mid"))
    else:
        files = sorted(Path(".").glob(args.input))
    
    if not files:
        print(f"❌ No MIDI files found: {args.input}")
        return 1
    
    print(f"🎸 Evaluating {len(files)} bass MIDI files...")
    print(f"   Chord root: {args.chord_root} {args.mode}")
    
    # Evaluate each file
    results = []
    for f in files:
        result = file_metrics_bass(f, args.chord_root, args.mode)
        results.append(result)
        
        if result.get("valid"):
            print(f"  ✅ {f.name}")
        else:
            print(f"  ❌ {f.name}: {result.get('reason', 'unknown')}")
    
    # Aggregate
    aggregated = aggregate_metrics(results)
    
    # Check quality gates
    threshold_flags = []
    summary = aggregated.get("summary", {})
    for name, stats in summary.items():
        if stats and "mean" in stats:
            flag = _flag_metric(name, stats["mean"])
            if flag:
                threshold_flags.append(flag)
    
    # Provenance (git info)
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "unknown"
    
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_branch = "unknown"
    
    # Assemble output
    output = {
        "schema_version": SCHEMA_VERSION,
        "instrument": "bass",
        "evaluation_date": None,  # Will be set by CI
        "fileset_hash": _fileset_hash(files),
        "chord_root": args.chord_root,
        "mode": args.mode,
        "provenance": {
            "git_commit": git_commit,
            "git_branch": git_branch,
        },
        "thresholds": THRESHOLDS,
        "threshold_flags": threshold_flags,
        "aggregated": aggregated,
        "per_file": results,
    }
    
    # Write JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Aggregated metrics:")
    for name, stats in summary.items():
        if stats and "mean" in stats:
            print(f"   {name}: {stats['mean']:.3f} (±{stats['std']:.3f})")
    
    print(f"\n🚦 Quality Gates:")
    if threshold_flags:
        print(f"   ❌ FAIL - Violations: {', '.join(threshold_flags)}")
    else:
        print(f"   ✅ PASS - All metrics within thresholds")
    
    print(f"\n✅ Results written to: {out_json}")
    
    # Write CSV (optional)
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["file", "valid", "tempo", "time_signature", "bars", "note_count"]
            header.extend(THRESHOLDS.keys())
            writer.writerow(header)
            
            # Rows
            for r in results:
                if r.get("valid"):
                    row = [
                        r["file"],
                        r["valid"],
                        r["tempo"],
                        r["time_signature"],
                        r["bars"],
                        r["note_count"],
                    ]
                    metrics = r.get("metrics", {})
                    for name in THRESHOLDS.keys():
                        row.append(metrics.get(name, ""))
                    writer.writerow(row)
        
        print(f"✅ CSV written to: {out_csv}")
    
    # Exit code based on quality gates
    return 1 if threshold_flags else 0


if __name__ == "__main__":
    exit(main())
