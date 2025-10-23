#!/usr/bin/env python3
"""LAMDAのpickleファイル内容確認スクリプト"""
import pickle
from pathlib import Path

pickle_path = Path("/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3/data/Los-Angeles-MIDI/CHORDS_DATA/LAMDa_CHORDS_DATA_10000.pickle")

print(f"Loading: {pickle_path.name}")
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

print(f"\n{'='*60}")
print(f"Type: {type(data)}")
print(f"Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")

if isinstance(data, list) and len(data) > 0:
    print(f"\n{'='*60}")
    print(f"最初のエントリ:")
    first = data[0]
    print(f"  Type: {type(first)}")
    print(f"  Length: {len(first) if hasattr(first, '__len__') else 'N/A'}")
    
    if isinstance(first, list) and len(first) >= 2:
        print(f"\n  構造:")
        print(f"    [0] (file_id): {first[0]}")
        print(f"    [1] (events): {type(first[1])}, length={len(first[1]) if hasattr(first[1], '__len__') else 'N/A'}")
        if isinstance(first[1], list) and len(first[1]) > 0:
            print(f"    [1][0] (最初のevent): {first[1][0][:30] if len(first[1][0]) > 30 else first[1][0]}")
    
    print(f"\n{'='*60}")
    print(f"サンプル（最初の5エントリ）:")
    for i, item in enumerate(data[:5]):
        if isinstance(item, list) and len(item) >= 2:
            file_id = item[0]
            n_events = len(item[1]) if hasattr(item[1], '__len__') else '?'
            print(f"  [{i}]: file_id={file_id}, n_events={n_events}")
        else:
            print(f"  [{i}]: {type(item)}, len={len(item) if hasattr(item, '__len__') else 'N/A'}")

elif isinstance(data, dict):
    print(f"\n{'='*60}")
    print(f"辞書のキー（最初の10個）:")
    for k in list(data.keys())[:10]:
        print(f"  - {k}")
    
    print(f"\n{'='*60}")
    print(f"最初のエントリ詳細:")
    first_key = list(data.keys())[0]
    first_val = data[first_key]
    print(f"  Key: {first_key}")
    print(f"  Value type: {type(first_val)}")
    print(f"  Value: {str(first_val)[:300]}")

print(f"\n{'='*60}")
