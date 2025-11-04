#!/usr/bin/env python3
"""
最小bars.parquet生成（chordmap.jsonから）
"""
import json
import pandas as pd
from pathlib import Path

song_dir = Path(
    "/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/suno_ai/suno_themesong/song_001"
)
chordmap_path = song_dir / "analysis/chordmap.json"

# chordmap読み込み
chordmap_data = json.loads(chordmap_path.read_text(encoding="utf-8"))
events = chordmap_data.get("events", [])

# 最終時刻から小節数推定（仮定: 4/4拍子、4QL=1小節）
max_time = max((ev["time"] for ev in events), default=0)
num_bars = int(max_time / 4.0) + 1

# bars.parquet生成（最小構造）
bars_data = []
for bar_idx in range(num_bars):
    bars_data.append(
        {
            "bar": bar_idx,
            "start_beat": bar_idx * 4.0,
            "end_beat": (bar_idx + 1) * 4.0,
            "time_signature": "4/4",
            "tempo_bpm": 120.0,
            "energy": 0.5,
            "section_label": "verse",  # デフォルト
        }
    )

bars_df = pd.DataFrame(bars_data)
bars_df.to_parquet(song_dir / "bars.parquet")

print(f"✅ Generated bars.parquet: {len(bars_df)} bars")
print(f"   Max time: {max_time:.1f} QL, Bars: {num_bars}")
