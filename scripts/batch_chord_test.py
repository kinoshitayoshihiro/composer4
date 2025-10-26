#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/batch_chord_test.py

複数songでの大規模テスト・精度評価

Usage:
  python scripts/batch_chord_test.py --base data/suno_ai --output results/batch_test.json

機能:
- 複数songの自動コード認識実行
- 手動chordmap（sections.json）との精度比較
- 統計レポート生成（平均精度、キー差分頻度等）
"""
from __future__ import annotations
import argparse, json, sys, subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Copy from compare_chordmaps.py
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def parse_note(note_str: str) -> Tuple[int, str]:
    """Parse note string to (root_idx, quality)"""
    if note_str == "N":
        return (-1, "N")
    
    note_str = note_str.strip()
    if len(note_str) < 1:
        return (-1, "N")
    
    # Check for sharp
    if len(note_str) > 1 and note_str[1] in ['#', '♯']:
        root = note_str[:2]
        quality = note_str[2:]
    else:
        root = note_str[0]
        quality = note_str[1:]
    
    # Normalize
    if root not in NOTE_NAMES:
        return (-1, "N")
    
    root_idx = NOTE_NAMES.index(root)
    
    # Normalize quality
    quality = quality.lower().replace('maj', '').replace('minor', 'm').replace('min', 'm')
    if quality == '':
        quality = 'maj'
    
    return (root_idx, quality)

def transpose_note(note_str: str, semitones: int) -> str:
    """Transpose note by semitones"""
    if note_str == "N":
        return "N"
    
    root_idx, quality = parse_note(note_str)
    if root_idx == -1:
        return "N"
    
    new_root_idx = (root_idx + semitones) % 12
    new_root = NOTE_NAMES[new_root_idx]
    
    # Reconstruct chord
    if quality == 'maj':
        return new_root
    else:
        return f"{new_root}{quality}"

def parse_manual_sections(sections_path: Path) -> List[Tuple[float, str]]:
    """Parse manual sections.json to list of (ql, chord)"""
    if not sections_path.exists():
        return []
    
    try:
        data = json.loads(sections_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    
    # List format
    if isinstance(data, list):
        chords = []
        for sec in data:
            bar = sec.get("bar", 0)
            ql = bar * 4.0  # Assume 4/4
            chord = sec.get("chord", "N")
            chords.append((ql, chord))
        return sorted(chords, key=lambda x: x[0])
    
    # Dict format
    else:
        sections = data.get("sections", [])
        chords = []
        for sec in sections:
            bar = sec.get("bar", 0)
            ql = bar * 4.0
            chord = sec.get("chord", "N")
            chords.append((ql, chord))
        return sorted(chords, key=lambda x: x[0])

def parse_auto_chordmap(chordmap_path: Path) -> List[Tuple[float, str]]:
    """Parse auto-generated chordmap.json to list of (ql, chord)
    
    Supports both formats:
    - Legacy: [{"ql": 0.0, "chord": "Cmaj7"}, ...]
    - Unified: {"unit": "ql", "events": [{"time": 0.0, "root": "C", "quality": "maj7"}, ...]}
    """
    if not chordmap_path.exists():
        return []
    
    try:
        data = json.loads(chordmap_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    
    chords = []
    
    # Detect format
    if isinstance(data, dict) and "events" in data:
        # Unified format
        events = data["events"]
        for ev in events:
            ql = float(ev.get("time", 0))
            root = ev.get("root", "N")
            quality = ev.get("quality", "")
            
            # Convert to chord string
            if root == "N":
                chord = "N"
            else:
                # Map quality names
                quality_map = {
                    "maj": "",
                    "min": "m",
                    "maj7": "maj7",
                    "min7": "m7",
                    "dom7": "7",
                    "min7b5": "m7b5",
                    "sus4": "sus4",
                    "sus2": "sus2",
                    "add9": "add9",
                    "6": "6"
                }
                q = quality_map.get(quality, quality)
                chord = f"{root}{q}"
            
            chords.append((ql, chord))
    
    elif isinstance(data, list):
        # Legacy format
        for ev in data:
            ql = float(ev.get("ql", 0))
            chord = ev.get("chord", "N")
            chords.append((ql, chord))
    
    return sorted(chords, key=lambda x: x[0])

def find_closest_match(manual_chords: List[Tuple[float, str]], auto_chords: List[Tuple[float, str]], tolerance: float = 1.0) -> List[Tuple[str, str]]:
    """Find closest matches between manual and auto chords"""
    matches = []
    
    for man_ql, man_chord in manual_chords:
        # Find closest auto chord within tolerance
        best_dist = float('inf')
        best_auto = None
        
        for auto_ql, auto_chord in auto_chords:
            dist = abs(auto_ql - man_ql)
            if dist < best_dist and dist <= tolerance:
                best_dist = dist
                best_auto = auto_chord
        
        if best_auto is not None:
            matches.append((man_chord, best_auto))
    
    return matches

def evaluate_accuracy(manual_chords: List[Tuple[float, str]], auto_chords: List[Tuple[float, str]], tolerance: float = 1.0) -> Dict[str, float]:
    """Evaluate accuracy with optional transposition"""
    if not manual_chords or not auto_chords:
        return {
            "root_accuracy": 0.0,
            "quality_accuracy": 0.0,
            "full_accuracy": 0.0,
            "total_matches": 0,
            "best_transposition": 0
        }
    
    matches = find_closest_match(manual_chords, auto_chords, tolerance)
    
    if not matches:
        return {
            "root_accuracy": 0.0,
            "quality_accuracy": 0.0,
            "full_accuracy": 0.0,
            "total_matches": 0,
            "best_transposition": 0
        }
    
    # Try all transpositions (0-11 semitones)
    best_transposition = 0
    best_root_acc = 0.0
    
    for semitones in range(12):
        root_correct = 0
        
        for man_chord, auto_chord in matches:
            transposed_auto = transpose_note(auto_chord, semitones)
            man_root, _ = parse_note(man_chord)
            trans_root, _ = parse_note(transposed_auto)
            
            if man_root == trans_root and man_root != -1:
                root_correct += 1
        
        root_acc = root_correct / len(matches)
        if root_acc > best_root_acc:
            best_root_acc = root_acc
            best_transposition = semitones
    
    # Evaluate with best transposition
    root_correct = 0
    quality_correct = 0
    full_correct = 0
    
    for man_chord, auto_chord in matches:
        transposed_auto = transpose_note(auto_chord, best_transposition)
        man_root, man_qual = parse_note(man_chord)
        trans_root, trans_qual = parse_note(transposed_auto)
        
        if man_root == trans_root and man_root != -1:
            root_correct += 1
        
        if man_qual == trans_qual:
            quality_correct += 1
        
        if man_root == trans_root and man_qual == trans_qual and man_root != -1:
            full_correct += 1
    
    return {
        "root_accuracy": root_correct / len(matches),
        "quality_accuracy": quality_correct / len(matches),
        "full_accuracy": full_correct / len(matches),
        "total_matches": len(matches),
        "best_transposition": best_transposition
    }

def find_all_songs(base_dir: Path) -> List[Path]:
    """Find all song directories containing stems/ or stemswav_*"""
    songs = []
    for song_dir in sorted(base_dir.rglob("song_*")):
        if song_dir.is_dir():
            # Check for stems/ or stemswav_*
            stems_dir = song_dir / "stems"
            if not stems_dir.exists():
                # Try stemswav_*
                stemswav_dirs = list(song_dir.glob("stemswav_*"))
                if stemswav_dirs:
                    stems_dir = stemswav_dirs[0]
            
            if stems_dir.exists() and any(stems_dir.glob("*.wav")):
                songs.append(song_dir)
    return songs

def run_chord_recognition(song_dir: Path, output_dir: Path, force_key: Optional[str] = None, use_7th: bool = False) -> bool:
    """Run chord recognition on a song"""
    stems_dir = song_dir / "stems"
    if not stems_dir.exists():
        # Try stemswav_*
        stemswav_dirs = list(song_dir.glob("stemswav_*"))
        if stemswav_dirs:
            stems_dir = stemswav_dirs[0]
    
    sections_path = song_dir / "analysis" / "sections.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    chordmap_path = output_dir / "chordmap_auto.json"
    
    # Build command
    script = "ops/stem_harmony_7th.py" if use_7th else "ops/stem_harmony.py"
    cmd = [
        "python", script,
        "--stems", str(stems_dir),
        "--out", str(chordmap_path),
        "--exclude", "Vocals"
    ]
    
    if sections_path.exists():
        cmd += ["--sections", str(sections_path)]
    
    if force_key:
        cmd += ["--force-key", force_key]
    
    # Run
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minutes
        if result.returncode == 0:
            print(f"  ✓ Generated: {chordmap_path}")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def test_song(song_dir: Path, output_dir: Path, tolerance: float = 2.0, force_key: Optional[str] = None, use_7th: bool = False) -> Optional[Dict]:
    """Test single song and return results"""
    print(f"\n[Testing] {song_dir.name}")
    
    # Run chord recognition
    success = run_chord_recognition(song_dir, output_dir, force_key=force_key, use_7th=use_7th)
    if not success:
        return None
    
    # Load manual and auto chordmaps
    sections_path = song_dir / "analysis" / "sections.json"
    chordmap_path = output_dir / "chordmap_auto.json"
    
    manual_chords = parse_manual_sections(sections_path)
    auto_chords = parse_auto_chordmap(chordmap_path)
    
    if not manual_chords:
        print("  ⚠ No manual chordmap (sections.json)")
        return None
    
    # Evaluate accuracy
    metrics = evaluate_accuracy(manual_chords, auto_chords, tolerance)
    
    print(f"  Root accuracy: {metrics['root_accuracy']*100:.1f}%")
    print(f"  Quality accuracy: {metrics['quality_accuracy']*100:.1f}%")
    print(f"  Full accuracy: {metrics['full_accuracy']*100:.1f}%")
    print(f"  Best transposition: {metrics['best_transposition']} semitones")
    print(f"  Total matches: {metrics['total_matches']}")
    
    return {
        "song": song_dir.name,
        "metrics": metrics,
        "manual_events": len(manual_chords),
        "auto_events": len(auto_chords)
    }

def print_summary(results: List[Dict]):
    """Print summary statistics"""
    if not results:
        print("\n[Summary] No results to summarize")
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Average metrics
    avg_root = np.mean([r["metrics"]["root_accuracy"] for r in results])
    avg_quality = np.mean([r["metrics"]["quality_accuracy"] for r in results])
    avg_full = np.mean([r["metrics"]["full_accuracy"] for r in results])
    
    print(f"\nAverage Accuracy (n={len(results)} songs):")
    print(f"  Root:    {avg_root*100:.1f}%")
    print(f"  Quality: {avg_quality*100:.1f}%")
    print(f"  Full:    {avg_full*100:.1f}%")
    
    # Transposition distribution
    transpositions = [r["metrics"]["best_transposition"] for r in results]
    unique_trans, counts = np.unique(transpositions, return_counts=True)
    
    print(f"\nKey Difference Distribution:")
    for trans, count in sorted(zip(unique_trans, counts), key=lambda x: -x[1]):
        print(f"  {trans:+2d} semitones: {count} songs ({count/len(results)*100:.1f}%)")
    
    # Best and worst songs
    results_sorted = sorted(results, key=lambda x: x["metrics"]["root_accuracy"], reverse=True)
    
    print(f"\nBest 3 Songs (Root Accuracy):")
    for r in results_sorted[:3]:
        print(f"  {r['song']}: {r['metrics']['root_accuracy']*100:.1f}%")
    
    print(f"\nWorst 3 Songs (Root Accuracy):")
    for r in results_sorted[-3:]:
        print(f"  {r['song']}: {r['metrics']['root_accuracy']*100:.1f}%")

def main():
    ap = argparse.ArgumentParser(description="Batch chord recognition test for multiple songs")
    ap.add_argument("--base", required=True, help="Base directory containing song_* folders")
    ap.add_argument("--output", required=True, help="Output JSON file for results")
    ap.add_argument("--tolerance", type=float, default=2.0, help="QL tolerance for matching (default: 2.0)")
    ap.add_argument("--force-key", help="Force key for all songs (e.g., 'C', 'Am')")
    ap.add_argument("--use-7th", action="store_true", help="Use 7th chords recognition (stem_harmony_7th.py)")
    ap.add_argument("--max-songs", type=int, help="Maximum number of songs to test")
    args = ap.parse_args()
    
    base_dir = Path(args.base)
    output_path = Path(args.output)
    
    if not base_dir.exists():
        print(f"[ERROR] Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all songs
    songs = find_all_songs(base_dir)
    
    if not songs:
        print(f"[ERROR] No songs found in {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.max_songs:
        songs = songs[:args.max_songs]
    
    print(f"[INFO] Found {len(songs)} songs to test")
    
    # Test each song
    results = []
    output_base = output_path.parent / "batch_test_outputs"
    
    for song_dir in songs:
        song_output = output_base / song_dir.name
        result = test_song(song_dir, song_output, tolerance=args.tolerance, force_key=args.force_key, use_7th=args.use_7th)
        if result:
            results.append(result)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "total_songs": len(songs),
            "successful_tests": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_path}")
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()
