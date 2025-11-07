#!/usr/bin/env python3
"""
GMD (Groove MIDI Dataset) Preprocessing
========================================
Phase B-1: GMD MIDI-only前処理

入力:
  - data/Magenta_Studio/datasets/GMD/groove/drummer*/session*/*.mid

出力:
  - data/GMD_processed/
      - index.parquet（全MIDIファイルのメタデータ）
      - groove_stats.json（Groove指標統計）
      - train.txt, val.txt, test.txt（データ分割リスト）

Usage:
  python ops/gmd_preprocess.py \\
    --gmd-root data/Magenta_Studio/datasets/GMD/groove \\
    --out-dir data/GMD_processed \\
    --val-ratio 0.1 \\
    --test-ratio 0.1
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    print("⚠️  pretty_midi not installed", file=sys.stderr)


def parse_filename(filename: str) -> Dict:
    """
    GMD MIDIファイル名パース
    
    例: "1_funk_80_beat_4-4.mid"
    → {"id": 1, "genre": "funk", "bpm": 80, "type": "beat", "time_sig": "4-4"}
    """
    pattern = r'(\d+)_([a-z\-]+)_(\d+)_(beat|fill)_(\d+\-\d+)\.mid'
    match = re.match(pattern, filename)
    
    if not match:
        return {}
    
    return {
        "id": int(match.group(1)),
        "genre": match.group(2),
        "bpm": int(match.group(3)),
        "type": match.group(4),
        "time_sig": match.group(5)
    }


def extract_groove_stats(midi_path: Path) -> Dict:
    """
    Groove指標抽出（Velocity std, IOI std, Note density）
    
    Returns:
        {
            "velocity_std": float,
            "ioi_std": float,
            "note_density": float (notes/sec),
            "duration": float (sec),
            "num_notes": int
        }
    """
    if not PRETTY_MIDI_AVAILABLE:
        return {}
    
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        drums = [inst for inst in pm.instruments if inst.is_drum]
        
        if not drums:
            return {}
        
        notes = [n for inst in drums for n in inst.notes]
        
        if not notes:
            return {}
        
        # Velocity std
        vels = np.array([n.velocity for n in notes])
        vel_std = float(vels.std())
        
        # IOI std (Inter-Onset Interval)
        onsets = np.array([n.start for n in notes])
        if len(onsets) > 1:
            ioi = np.diff(np.sort(onsets))
            ioi_std = float(ioi.std())
        else:
            ioi_std = 0.0
        
        # Note density
        duration = pm.get_end_time()
        note_density = len(notes) / duration if duration > 0 else 0.0
        
        return {
            "velocity_std": vel_std,
            "ioi_std": ioi_std,
            "note_density": note_density,
            "duration": duration,
            "num_notes": len(notes)
        }
    except Exception as e:
        print(f"⚠️  Failed to process {midi_path}: {e}", file=sys.stderr)
        return {}


def scan_gmd_dataset(gmd_root: Path) -> pd.DataFrame:
    """
    GMDデータセットスキャン→DataFrame生成
    
    Returns:
        DataFrame with columns:
          - file_path, drummer, session, id, genre, bpm, type, time_sig,
            velocity_std, ioi_std, note_density, duration, num_notes
    """
    rows = []
    
    for drummer_dir in sorted(gmd_root.glob("drummer*")):
        if not drummer_dir.is_dir():
            continue
        
        drummer = drummer_dir.name
        
        for session_dir in sorted(drummer_dir.glob("session*")):
            if not session_dir.is_dir():
                continue
            
            session = session_dir.name
            
            for midi_path in sorted(session_dir.glob("*.mid")):
                # ファイル名パース
                meta = parse_filename(midi_path.name)
                
                if not meta:
                    continue
                
                # Groove指標抽出
                stats = extract_groove_stats(midi_path)
                
                row = {
                    "file_path": str(midi_path.relative_to(gmd_root)),
                    "drummer": drummer,
                    "session": session,
                    **meta,
                    **stats
                }
                
                rows.append(row)
    
    return pd.DataFrame(rows)


def split_dataset(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    データセット分割（drummer単位、評価漏れ防止）
    
    Returns:
        (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    
    # drummer単位で分割（eval_sessionは除外）
    drummers = [d for d in df["drummer"].unique() if "eval" not in d]
    np.random.shuffle(drummers)
    
    n_val = int(len(drummers) * val_ratio)
    n_test = int(len(drummers) * test_ratio)
    
    val_drummers = set(drummers[:n_val])
    test_drummers = set(drummers[n_val:n_val+n_test])
    train_drummers = set(drummers[n_val+n_test:])
    
    train_df = df[df["drummer"].isin(train_drummers)]
    val_df = df[df["drummer"].isin(val_drummers)]
    test_df = df[df["drummer"].isin(test_drummers)]
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="GMD MIDI Preprocessing")
    parser.add_argument(
        "--gmd-root",
        type=Path,
        default=Path("data/Magenta_Studio/datasets/GMD/groove"),
        help="GMDデータセットルートディレクトリ"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/GMD_processed"),
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    if not PRETTY_MIDI_AVAILABLE:
        print("❌ pretty_midi required. Install with: pip install pretty_midi")
        sys.exit(1)
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 GMD Dataset Preprocessing")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"GMD root: {args.gmd_root}")
    print(f"Output  : {args.out_dir}")
    print()
    
    # 1. スキャン
    print("🔍 Scanning GMD dataset...")
    df = scan_gmd_dataset(args.gmd_root)
    print(f"   Found: {len(df)} MIDI files")
    print()
    
    # 2. データセット分割
    print("✂️  Splitting dataset...")
    train_df, val_df, test_df = split_dataset(
        df, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"   Train: {len(train_df)} files")
    print(f"   Val  : {len(val_df)} files")
    print(f"   Test : {len(test_df)} files")
    print()
    
    # 3. Groove指標統計
    print("📈 Computing groove statistics...")
    groove_stats = {
        "train": {
            "velocity_std": {
                "mean": float(train_df["velocity_std"].mean()),
                "std": float(train_df["velocity_std"].std()),
                "min": float(train_df["velocity_std"].min()),
                "max": float(train_df["velocity_std"].max())
            },
            "ioi_std": {
                "mean": float(train_df["ioi_std"].mean()),
                "std": float(train_df["ioi_std"].std()),
                "min": float(train_df["ioi_std"].min()),
                "max": float(train_df["ioi_std"].max())
            },
            "note_density": {
                "mean": float(train_df["note_density"].mean()),
                "std": float(train_df["note_density"].std()),
                "min": float(train_df["note_density"].min()),
                "max": float(train_df["note_density"].max())
            }
        }
    }
    print(f"   Velocity std: {groove_stats['train']['velocity_std']['mean']:.2f} ± {groove_stats['train']['velocity_std']['std']:.2f}")
    print(f"   IOI std     : {groove_stats['train']['ioi_std']['mean']:.4f} ± {groove_stats['train']['ioi_std']['std']:.4f}")
    print(f"   Note density: {groove_stats['train']['note_density']['mean']:.2f} ± {groove_stats['train']['note_density']['std']:.2f} notes/sec")
    print()
    
    # 4. 保存
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # index.parquet
    index_path = args.out_dir / "index.parquet"
    df.to_parquet(index_path, index=False)
    print(f"✅ Saved: {index_path} ({len(df)} files)")
    
    # groove_stats.json
    stats_path = args.out_dir / "groove_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(groove_stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {stats_path}")
    
    # train/val/test.txt
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_path = args.out_dir / f"{name}.txt"
        split_df["file_path"].to_csv(split_path, index=False, header=False)
        print(f"✅ Saved: {split_path} ({len(split_df)} files)")
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ GMD Preprocessing Complete!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
