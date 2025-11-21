#!/usr/bin/env python3
"""
Unified Metrics Generator - Phase 124 (P1-3)

全Phase共通のメトリクスJSONを生成し、Phase間比較を可能にする。

メトリクス形式:
{
  "phase": 124,
  "song_id": "song_001",
  "plan_events": {"piano": 779, "guitar": 1263, ..., "total": 11488},
  "midi_notes": {"piano": 703, "guitar": 874, ..., "total": 6007},
  "oaf": {"notes_total": 3047, "applied_events": 779, "coverage_pct": 100.0},
  "crepe": {"frames": 47997, "coverage_sec": 479.97},
  "ci": {"pass": true, "checks": {...}},
  "versions": {"oaf_mapper": "0.1.0", "midi_writer": ">=0.3.0"},
  "generated_at": "2025-11-08T18:42:00+09:00"
}

使い方:
  python scripts/unified_metrics_generator.py \
    --phase 124 \
    --song-dir data/suno_ai/suno_themesong/song_001 \
    --plan full_arrangement_phase124.json \
    --midi full_arrangement_phase124.mid \
    --oaf-report oaf_dynamics_report.json \
    --ci-report ci_report.json \
    --out unified_metrics_phase124.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

def count_plan_events(plan_path: Path) -> Dict[str, int]:
    """
    Plan JSONからパート別イベント数をカウント
    
    Returns:
        {part_name: event_count, "total": total_count}
    """
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    counts = {}
    total = 0
    
    if "tracks" in plan:
        for track in plan["tracks"]:
            part_name = track.get("role") or track.get("name", "unknown")
            event_count = len(track.get("events", []))
            counts[part_name] = event_count
            total += event_count
    else:
        # 旧形式（Bass, Guitar等直接）
        for part in ["bass", "guitar", "piano", "strings", "drums"]:
            if part in plan:
                event_count = len(plan[part])
                counts[part] = event_count
                total += event_count
    
    counts["total"] = total
    return counts

def count_midi_notes(midi_path: Path) -> Dict[str, int]:
    """
    MIDIファイルからトラック別ノート数をカウント
    
    Returns:
        {track_name: note_count, "total": total_count}
    """
    if not pretty_midi:
        return {"error": "pretty_midi not available", "total": 0}
    
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    counts = {}
    total = 0
    
    for inst in pm.instruments:
        track_name = (inst.name or "Unnamed")[:32].lower()
        note_count = len(inst.notes)
        counts[track_name] = note_count
        total += note_count
    
    counts["total"] = total
    return counts

def extract_oaf_metrics(oaf_report_path: Optional[Path]) -> Dict[str, Any]:
    """
    OaF dynamics reportからメトリクス抽出
    
    Returns:
        {notes_total, applied_events, coverage_pct}
    """
    if not oaf_report_path or not oaf_report_path.exists():
        return {"notes_total": 0, "applied_events": 0, "coverage_pct": 0.0}
    
    report = json.loads(oaf_report_path.read_text(encoding="utf-8"))
    notes_total = report.get("oaf_notes", 0)
    applied = report.get("applied", 0)
    coverage = (applied / notes_total * 100) if notes_total > 0 else 0.0
    
    return {
        "notes_total": notes_total,
        "applied_events": applied,
        "coverage_pct": round(coverage, 1)
    }

def extract_crepe_metrics(song_dir: Path) -> Dict[str, Any]:
    """
    CREPE（Bass F0）メトリクス抽出
    
    Returns:
        {frames, coverage_sec}
    """
    bass_f0_path = song_dir / "bass_f0.parquet"
    if not bass_f0_path.exists():
        return {"frames": 0, "coverage_sec": 0.0}
    
    try:
        f0_df = pd.read_parquet(bass_f0_path)
        frames = len(f0_df)
        # 10ms刻み前提
        coverage_sec = round(frames * 0.01, 2)
        return {"frames": frames, "coverage_sec": coverage_sec}
    except Exception:
        return {"frames": 0, "coverage_sec": 0.0}

def extract_ci_metrics(ci_report_path: Optional[Path]) -> Dict[str, Any]:
    """
    CI reportからメトリクス抽出
    
    Returns:
        {pass: bool, checks: {...}}
    """
    if not ci_report_path or not ci_report_path.exists():
        return {"pass": False, "checks": {}}
    
    report = json.loads(ci_report_path.read_text(encoding="utf-8"))
    
    # CI結果集計
    results = report.get("results", [])
    checks = {}
    all_pass = True
    
    for r in results:
        name = r.get("name", "unknown")
        status = r.get("status", "fail")
        checks[name] = (status == "pass")
        if status != "pass":
            all_pass = False
    
    return {
        "pass": all_pass,
        "checks": checks
    }

def generate_unified_metrics(
    phase: int,
    song_dir: Path,
    plan_path: Optional[Path],
    midi_path: Optional[Path],
    oaf_report_path: Optional[Path],
    ci_report_path: Optional[Path]
) -> Dict[str, Any]:
    """
    統一メトリクスJSON生成
    """
    song_id = song_dir.name
    
    # Plan events
    plan_events = count_plan_events(plan_path) if plan_path and plan_path.exists() else {}
    
    # MIDI notes
    midi_notes = count_midi_notes(midi_path) if midi_path and midi_path.exists() else {}
    
    # OaF
    oaf = extract_oaf_metrics(oaf_report_path)
    
    # CREPE
    crepe = extract_crepe_metrics(song_dir)
    
    # CI
    ci = extract_ci_metrics(ci_report_path)
    
    # Versions
    versions = {
        "oaf_mapper": "0.1.0",
        "midi_writer": ">=0.3.0",
        "plan_schema_check": ">=0.1.0"
    }
    
    # Unified metrics
    metrics = {
        "phase": phase,
        "song_id": song_id,
        "plan_events": plan_events,
        "midi_notes": midi_notes,
        "oaf": oaf,
        "crepe": crepe,
        "ci": ci,
        "versions": versions,
        "generated_at": datetime.now().astimezone().isoformat()
    }
    
    return metrics

def main():
    ap = argparse.ArgumentParser(description="Unified Metrics Generator - Phase 124")
    ap.add_argument("--phase", type=int, required=True, help="Phase number")
    ap.add_argument("--song-dir", required=True, help="Song directory")
    ap.add_argument("--plan", help="Plan JSON (optional)")
    ap.add_argument("--midi", help="MIDI file (optional)")
    ap.add_argument("--oaf-report", help="OaF dynamics report JSON (optional)")
    ap.add_argument("--ci-report", help="CI report JSON (optional)")
    ap.add_argument("--out", required=True, help="Output unified metrics JSON")
    args = ap.parse_args()
    
    song_dir = Path(args.song_dir)
    plan_path = Path(args.plan) if args.plan else None
    midi_path = Path(args.midi) if args.midi else None
    oaf_report_path = Path(args.oaf_report) if args.oaf_report else None
    ci_report_path = Path(args.ci_report) if args.ci_report else None
    
    # Generate metrics
    metrics = generate_unified_metrics(
        args.phase,
        song_dir,
        plan_path,
        midi_path,
        oaf_report_path,
        ci_report_path
    )
    
    # Output
    out_path = Path(args.out)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n✅ Unified Metrics Generated (Phase {args.phase}):")
    print(f"   Song: {metrics['song_id']}")
    print(f"   Plan events: {metrics['plan_events'].get('total', 0)}")
    print(f"   MIDI notes: {metrics['midi_notes'].get('total', 0)}")
    print(f"   OaF coverage: {metrics['oaf']['coverage_pct']}%")
    print(f"   CI pass: {metrics['ci']['pass']}")
    print(f"   Saved to: {out_path}")

if __name__ == "__main__":
    main()
