#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シンプル版chordmap生成テスト (stem_harmony.py使用)
トライアドのみ抽出、複雑なコードは手動で追加
"""

# ⚠️ 環境対策: numba Bus error回避（最優先で設定）
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"

import sys
import platform
import json
from pathlib import Path

print("=" * 60)
print("🔧 Environment Diagnostic")
print("=" * 60)

# バージョン確認
print(f"Python: {sys.version}")
print(f"Arch  : {platform.machine()}")

try:
    import numpy as np
    print(f"numpy     : {np.__version__}")
except Exception as e:
    print(f"numpy     : ERROR - {e}")
    sys.exit(1)

try:
    import numba
    print(f"numba     : {numba.__version__}")
except Exception as e:
    print(f"numba     : ERROR or JIT disabled - {e}")

try:
    import llvmlite
    print(f"llvmlite  : {llvmlite.__version__}")
except Exception as e:
    print(f"llvmlite  : ERROR - {e}")

try:
    import librosa
    print(f"librosa   : {librosa.__version__}")
except Exception as e:
    print(f"librosa   : ERROR - {e}")
    sys.exit(1)

print()
print("=" * 60)
print("🎯 Running stem_harmony.py (Simple HMM version)")
print("=" * 60)

# シンプル版で実行（HMMベース、トライアドのみ）
cmd = [
    sys.executable,
    "ops/stem_harmony.py",
    "--stems", "data/suno_ai/suno_themesong/song_001/stemswav_001",
    "--out", "data/suno_ai/suno_themesong/song_001/analysis/chordmap_simple.json",
    "--sections", "data/suno_ai/suno_themesong/song_001/analysis/sections.json",
    "--exclude", "Vocals",
    "--exclude", "Backing Vocals",
    "--exclude", "Drums",
    "--exclude", "Percussion",
    "--stay", "0.85",  # 少し低めに設定（コード変化を促す）
    "--near", "0.05",  # 近傍遷移確率を少し高めに
    "--emit-confidence",
    "--min-dwell-ql", "4.0"  # 最低1小節（4QL）保持
]

print(f"Command: {' '.join(cmd)}")
print()

# ⚠️ JIT無効化を子プロセスに確実に伝える
import subprocess
env = os.environ.copy()
env["NUMBA_DISABLE_JIT"] = "1"              # JIT完全オフ（遅くなるが安定）
env["NUMBA_THREADING_LAYER"] = "workqueue"  # 競合回避（念のため）
env["NUMBA_CACHE_DIR"] = "/tmp/numba-cache" # 壊れキャッシュを避ける

result = subprocess.run(cmd, capture_output=True, text=True, env=env)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode != 0:
    print(f"\n❌ Failed with exit code {result.returncode}")
    sys.exit(result.returncode)

print()
print("=" * 60)
print("📊 Analyzing Results")
print("=" * 60)

# 結果分析
output_path = Path("data/suno_ai/suno_themesong/song_001/analysis/chordmap_simple.json")
if output_path.exists():
    with output_path.open("r") as f:
        data = json.load(f)
    
    events = data.get("events", [])
    if events:
        total_bars = int(events[-1]["time"] / 4.0)
        density = len(events) / max(total_bars, 1)
        
        print(f"✅ Success!")
        print(f"Total events: {len(events)}")
        print(f"Total duration: {events[-1]['time']}QL = {total_bars} bars")
        print(f"Density: {len(events)} / {total_bars} = {density:.3f} chords/bar")
        print(f"Target: 0.9-1.2 chords/bar")
        
        if 0.9 <= density <= 1.2:
            print("🎉 Density is in target range!")
        elif density < 0.9:
            print("⚠️ Density is lower than target (chords change too slowly)")
        else:
            print("⚠️ Density is higher than target (chords change too quickly)")
        
        # コード分布
        from collections import Counter
        chord_labels = [f"{ev['root']}{ev['quality']}" for ev in events]
        counter = Counter(chord_labels)
        print()
        print("Top 10 chords:")
        for chord, count in counter.most_common(10):
            print(f"  {chord}: {count} ({count/len(events)*100:.1f}%)")
        
        # トライアドの比率
        triad_count = sum(1 for q in [e['quality'] for e in events] if q in ('', 'm'))
        extended_count = len(events) - triad_count
        print()
        print(f"Triads (maj/min): {triad_count} ({triad_count/len(events)*100:.1f}%)")
        print(f"Extended chords: {extended_count} ({extended_count/len(events)*100:.1f}%)")
        
        if triad_count / len(events) > 0.8:
            print("✅ Good! Mostly triads (suitable for manual refinement)")
    else:
        print("❌ No events in output")
else:
    print(f"❌ Output file not found: {output_path}")
