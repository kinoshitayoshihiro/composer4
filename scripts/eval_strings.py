#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strings Generator Evaluation Script (Phase 4.6)

Evaluates Strings Generator output:
- legato_connection_rate: レガート連結率
- leap_rate: 跳躍の割合
- max_leap_semitones: 最大跳躍幅
- chord_spread_semitones: 和声音の広がり
- velocity_std: ベロシティ分散
- bar_violation_rate: 小節境界逸脱率

Usage:
    python scripts/eval_strings.py \\
      --input output/strings/*.mid \\
      --out-json output/reports/strings_eval.json \\
      --out-csv output/reports/strings_eval.csv
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

# Strings range constants (from eval_drum_batch_stratified.py)
STRINGS_RANGE_MIN = 43  # G2
STRINGS_RANGE_MAX = 88  # E6

# Threshold definitions (aligned with config/quality_gates.yaml)
THRESHOLDS = {
    "legato_connection_rate": {"min": 0.60, "direction": "higher_is_better"},
    "leap_rate": {"max": 0.15, "direction": "lower_is_better"},
    "max_leap_semitones": {"max": 12, "direction": "lower_is_better"},
    "chord_spread_semitones": {"max": 24, "direction": "lower_is_better"},
    "velocity_std": {"min": 12.0, "direction": "higher_is_better"},
    "bar_violation_rate": {"max": 0.02, "direction": "lower_is_better"},
}


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


def file_metrics_strings(mid_path: Path) -> Dict[str, Any]:
    """
    Calculate strings metrics for a single MIDI file.
    
    Args:
        mid_path: Path to MIDI file
    
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
    
    # Find strings track (non-drum, within strings range)
    strings_track = None
    for inst in pm.instruments:
        if not inst.is_drum and inst.notes:
            # Check if most notes are in strings range
            strings_notes = [n for n in inst.notes if STRINGS_RANGE_MIN <= n.pitch <= STRINGS_RANGE_MAX]
            if len(strings_notes) >= len(inst.notes) * 0.5:
                strings_track = inst
                break
    
    if not strings_track or not strings_track.notes:
        return {
            "file": str(mid_path.name),
            "error": "no_strings_notes",
            "valid": False,
            "reason": "no_strings_tracks_or_notes"
        }
    
    notes = sorted(strings_track.notes, key=lambda n: n.start)
    
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
    
    # 1) legato_connection_rate
    # Notes are "legato connected" if next note starts within 50ms of previous note end
    legato_threshold_ms = 50.0
    legato_connections = 0
    
    for i in range(len(notes) - 1):
        gap = (notes[i+1].start - notes[i].end) * 1000.0  # ms
        if abs(gap) <= legato_threshold_ms:
            legato_connections += 1
    
    legato_connection_rate = legato_connections / (len(notes) - 1) if len(notes) > 1 else 0.0
    
    # 2) leap_rate & max_leap_semitones
    # Calculate intervals between consecutive notes (melodic line)
    leaps = []
    for i in range(1, len(notes)):
        interval = abs(notes[i].pitch - notes[i-1].pitch)
        leaps.append(interval)
    
    leap_threshold = 3  # Major 3rd or larger
    leap_count = sum(1 for leap in leaps if leap >= leap_threshold)
    leap_rate = leap_count / len(leaps) if leaps else 0.0
    max_leap_semitones = max(leaps) if leaps else 0
    
    # 3) chord_spread_semitones
    # Find simultaneous notes (within 50ms window) and measure pitch spread
    time_windows = []
    current_window = [notes[0]]
    
    for i in range(1, len(notes)):
        if notes[i].start - current_window[0].start <= 0.05:  # 50ms window
            current_window.append(notes[i])
        else:
            if len(current_window) > 1:
                time_windows.append(current_window)
            current_window = [notes[i]]
    
    if len(current_window) > 1:
        time_windows.append(current_window)
    
    # Calculate spread for each chord window
    spreads = []
    for window in time_windows:
        pitches = [n.pitch for n in window]
        spread = max(pitches) - min(pitches)
        spreads.append(spread)
    
    chord_spread_semitones = max(spreads) if spreads else 0
    
    # 4) velocity_std
    vels = [n.velocity for n in notes]
    velocity_std = statistics.stdev(vels) if len(vels) > 1 else 0.0
    
    # 5) bar_violation_rate
    # Notes that span across bar boundaries
    violations = 0
    for n in notes:
        start_bar = int(n.start / bar_length)
        end_bar = int(n.end / bar_length)
        if start_bar != end_bar:
            violations += 1
    
    bar_violation_rate = violations / len(notes) if notes else 0.0
    
    # Assemble result
    metrics = {
        "legato_connection_rate": round(legato_connection_rate, 3),
        "leap_rate": round(leap_rate, 3),
        "max_leap_semitones": max_leap_semitones,
        "chord_spread_semitones": chord_spread_semitones,
        "velocity_std": round(velocity_std, 2),
        "bar_violation_rate": round(bar_violation_rate, 3),
    }
    
    return {
        "file": str(mid_path.name),
        "valid": True,
        "tempo": round(tempo, 1),
        "time_signature": tsig,
        "bars": bars,
        "note_count": len(notes),
        "chord_windows": len(time_windows),
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
    parser = argparse.ArgumentParser(description="Evaluate Strings Generator output")
    parser.add_argument("--input", required=True, help="Input MIDI files (glob pattern)")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--out-csv", help="Output CSV path (optional)")
    
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
    
    print(f"🎻 Evaluating {len(files)} strings MIDI files...")
    
    # Evaluate each file
    results = []
    for f in files:
        result = file_metrics_strings(f)
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
        "instrument": "strings",
        "evaluation_date": None,  # Will be set by CI
        "fileset_hash": _fileset_hash(files),
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
            header = ["file", "valid", "tempo", "time_signature", "bars", "note_count", "chord_windows"]
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
                        r["chord_windows"],
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
