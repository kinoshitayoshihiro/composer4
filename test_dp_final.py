#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
環境診断 + DP版chordmap生成テスト（ChatGPT推奨設定）
"""

# ⚠️ 環境対策: numba Bus error回避（最優先で設定）
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"

# さらにimport librosaの前に確実に設定
import sys
if "librosa" not in sys.modules:
    # librosaがまだimportされていないことを確認
    pass

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
print("🎯 Running stem_harmony_bar_level.py (DP version)")
print("=" * 60)

# DP版で実行
cmd = [
    sys.executable,
    "ops/stem_harmony_bar_level.py",
    "--stems", "data/suno_ai/suno_themesong/song_001/stemswav_001",
    "--out", "data/suno_ai/suno_themesong/song_001/analysis/chordmap_dp_final.json",
    "--sections", "data/suno_ai/suno_themesong/song_001/analysis/sections.json",
    "--exclude", "Vocals",
    "--exclude", "Backing Vocals",
    "--exclude", "Drums",
    "--exclude", "Percussion",
    "--use-dp",
    "--change-penalty", "0.15",
    "--emit-confidence",
    "--min-dwell-bars", "1"
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
output_path = Path("data/suno_ai/suno_themesong/song_001/analysis/chordmap_dp_final.json")
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
        else:
            print("⚠️ Density is outside target range")
        
        # コード分布
        from collections import Counter
        chord_labels = [f"{ev['root']}{ev['quality']}" for ev in events]
        counter = Counter(chord_labels)
        print()
        print("Top 10 chords:")
        for chord, count in counter.most_common(10):
            print(f"  {chord}: {count} ({count/len(events)*100:.1f}%)")
    else:
        print("❌ No events in output")
else:
    print(f"❌ Output file not found: {output_path}")
