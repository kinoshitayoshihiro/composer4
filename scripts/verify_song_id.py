#!/usr/bin/env python3
"""
song_idフィールド追加の簡易検証

Purpose:
- ComparisonResultにsong_idフィールドが正しく追加されたか確認

Usage:
    python scripts/verify_song_id.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from ml.traffic_splitter import ComparisonResult
from dataclasses import fields

def main():
    print("=" * 70)
    print("song_id フィールド追加検証")
    print("=" * 70)
    
    # ComparisonResultのフィールド一覧を取得
    print("\n� ComparisonResult fields:")
    required_metadata = ['run_id', 'git_sha', 'v3_model_sha256', 'v1_model_sha256', 'song_id']
    
    result_fields = [f.name for f in fields(ComparisonResult)]
    
    print(f"\n   Total fields: {len(result_fields)}")
    print(f"\n   Metadata fields:")
    for field in required_metadata:
        if field in result_fields:
            print(f"      ✅ {field}")
        else:
            print(f"      ❌ {field} (MISSING)")
    
    # 全フィールドを表示
    print(f"\n   All fields (first 15):")
    for i, field in enumerate(result_fields[:15], 1):
        marker = "✅" if field in required_metadata else "  "
        print(f"      {marker} {i}. {field}")
    
    # 検証結果
    missing = [f for f in required_metadata if f not in result_fields]
    
    print("\n" + "=" * 70)
    if missing:
        print(f"❌ song_id フィールド追加検証: 失敗")
        print(f"   Missing fields: {missing}")
        sys.exit(1)
    else:
        print("✅ song_id フィールド追加検証: 成功")
        print("   All required metadata fields present")
    print("=" * 70)

if __name__ == "__main__":
    main()
