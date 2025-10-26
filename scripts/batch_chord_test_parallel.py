#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/batch_chord_test_parallel.py

並列処理版バッチテスト（高速化）

改善点:
- multiprocessing.Pool で並列処理
- tqdm進捗表示
- cached版stem_harmonyを使用
"""
from __future__ import annotations
import argparse, json, sys, subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from tqdm import tqdm
    HAS_TQDM = True
except:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

# Reuse from batch_chord_test.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from batch_chord_test import (
    parse_note, transpose_note, parse_manual_sections, parse_auto_chordmap,
    find_closest_match, evaluate_accuracy, find_all_songs
)

def run_chord_recognition_worker(args_tuple: Tuple[Path, Path, Optional[str], bool]) -> Tuple[bool, Path]:
    """Worker function for parallel processing"""
    song_dir, output_dir, force_key, use_7th = args_tuple
    
    stems_dir = song_dir / "stems"
    if not stems_dir.exists():
        stemswav_dirs = list(song_dir.glob("stemswav_*"))
        if stemswav_dirs:
            stems_dir = stemswav_dirs[0]
    
    sections_path = song_dir / "analysis" / "sections.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    chordmap_path = output_dir / "chordmap_auto.json"
    
    # Use cached version for speed
    script = "ops/stem_harmony_7th.py" if use_7th else "ops/stem_harmony_cached.py"
    cmd = [
        sys.executable, script,
        "--stems", str(stems_dir),
        "--out", str(chordmap_path),
        "--exclude", "Vocals"
    ]
    
    if sections_path.exists():
        cmd += ["--sections", str(sections_path)]
    
    if force_key:
        cmd += ["--force-key", force_key]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        success = result.returncode == 0
        if not success:
            print(f"[ERROR] {song_dir.name}: {result.stderr[:200]}")
        return success, song_dir
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {song_dir.name}")
        return False, song_dir
    except Exception as e:
        print(f"[ERROR] {song_dir.name}: {e}")
        return False, song_dir

def test_song_parallel(song_dir: Path, output_dir: Path, tolerance: float = 2.0) -> Optional[Dict]:
    """Evaluate single song (after recognition is done)"""
    sections_path = song_dir / "analysis" / "sections.json"
    chordmap_path = output_dir / "chordmap_auto.json"
    
    manual_chords = parse_manual_sections(sections_path)
    auto_chords = parse_auto_chordmap(chordmap_path)
    
    if not manual_chords:
        return None
    
    metrics = evaluate_accuracy(manual_chords, auto_chords, tolerance)
    
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
    
    avg_root = np.mean([r["metrics"]["root_accuracy"] for r in results])
    avg_quality = np.mean([r["metrics"]["quality_accuracy"] for r in results])
    avg_full = np.mean([r["metrics"]["full_accuracy"] for r in results])
    
    print(f"\nAverage Accuracy (n={len(results)} songs):")
    print(f"  Root:    {avg_root*100:.1f}%")
    print(f"  Quality: {avg_quality*100:.1f}%")
    print(f"  Full:    {avg_full*100:.1f}%")
    
    transpositions = [r["metrics"]["best_transposition"] for r in results]
    unique_trans, counts = np.unique(transpositions, return_counts=True)
    
    print(f"\nKey Difference Distribution:")
    for trans, count in sorted(zip(unique_trans, counts), key=lambda x: -x[1]):
        print(f"  {trans:+2d} semitones: {count} songs ({count/len(results)*100:.1f}%)")
    
    results_sorted = sorted(results, key=lambda x: x["metrics"]["root_accuracy"], reverse=True)
    
    print(f"\nBest 3 Songs (Root Accuracy):")
    for r in results_sorted[:min(3, len(results))]:
        print(f"  {r['song']}: {r['metrics']['root_accuracy']*100:.1f}%")
    
    if len(results) > 3:
        print(f"\nWorst 3 Songs (Root Accuracy):")
        for r in results_sorted[-3:]:
            print(f"  {r['song']}: {r['metrics']['root_accuracy']*100:.1f}%")

def main():
    ap = argparse.ArgumentParser(description="Parallel batch chord recognition test")
    ap.add_argument("--base", required=True, help="Base directory containing song_* folders")
    ap.add_argument("--output", required=True, help="Output JSON file for results")
    ap.add_argument("--tolerance", type=float, default=2.0, help="QL tolerance for matching")
    ap.add_argument("--force-key", help="Force key for all songs")
    ap.add_argument("--use-7th", action="store_true", help="Use 7th chords recognition")
    ap.add_argument("--max-songs", type=int, help="Maximum number of songs to test")
    ap.add_argument("--workers", type=int, help=f"Number of parallel workers (default: {cpu_count()})")
    args = ap.parse_args()
    
    base_dir = Path(args.base)
    output_path = Path(args.output)
    
    if not base_dir.exists():
        print(f"[ERROR] Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    songs = find_all_songs(base_dir)
    
    if not songs:
        print(f"[ERROR] No songs found in {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.max_songs:
        songs = songs[:args.max_songs]
    
    n_workers = args.workers or cpu_count()
    print(f"[INFO] Found {len(songs)} songs to test")
    print(f"[INFO] Using {n_workers} parallel workers")
    
    output_base = output_path.parent / "batch_test_outputs"
    
    # Step 1: Run chord recognition in parallel
    print("\n[Phase 1/2] Running chord recognition...")
    
    tasks = []
    for song_dir in songs:
        song_output = output_base / song_dir.name
        tasks.append((song_dir, song_output, args.force_key, args.use_7th))
    
    with Pool(n_workers) as pool:
        if HAS_TQDM:
            results_recog = list(tqdm(pool.imap(run_chord_recognition_worker, tasks), total=len(tasks), desc="Recognition"))
        else:
            results_recog = pool.map(run_chord_recognition_worker, tasks)
    
    # Count successes
    successes = sum(1 for success, _ in results_recog if success)
    print(f"[INFO] Recognition completed: {successes}/{len(songs)} successful")
    
    # Step 2: Evaluate accuracy
    print("\n[Phase 2/2] Evaluating accuracy...")
    
    results = []
    eval_tasks = [(song_dir, output_base / song_dir.name, args.tolerance) 
                  for success, song_dir in results_recog if success]
    
    for song_dir, song_output, tolerance in (tqdm(eval_tasks, desc="Evaluation") if HAS_TQDM else eval_tasks):
        result = test_song_parallel(song_dir, song_output, tolerance)
        if result:
            results.append(result)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "total_songs": len(songs),
            "successful_tests": len(results),
            "workers": n_workers,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_path}")
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()
