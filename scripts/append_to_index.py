#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存のLAMDAインデックスに新しいレコードを追加

Usage:
    python scripts/append_to_index.py \\
        --existing output/drums_metadata/drums_metadata_v2.pickle \\
        --new output/drums_metadata_new/drums_metadata_v2.pickle
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent))
from cleaners.common import file_sha1, fileset_sha1


def append_to_index(
    existing_index_path: Path,
    new_index_path: Path,
) -> Path:
    """
    既存のインデックスに新しいインデックスのshardを追加
    
    手順:
    1. 既存インデックスを読み込み
    2. 新しいインデックスを読み込み
    3. 新しいshardをコピー（番号を振り直し）
    4. 既存インデックスのshards[]に追加
    5. 更新されたインデックスを保存
    
    Args:
        existing_index_path: 既存のインデックスpickleパス
        new_index_path: 新しいインデックスpickleパス
        
    Returns:
        更新されたインデックスのパス
    """
    # 1. 既存インデックス読み込み
    print(f"📖 Loading existing index: {existing_index_path}")
    with open(existing_index_path, "rb") as f:
        existing_index = pickle.load(f)
    
    existing_base_dir = Path(existing_index["base_dir"])
    existing_instrument = existing_index["instrument"]
    existing_shard_count = len(existing_index["shards"])
    existing_total_files = existing_index["total_files"]
    
    print(f"   Existing shards: {existing_shard_count}")
    print(f"   Existing files:  {existing_total_files}")
    
    # 2. 新しいインデックス読み込み
    print()
    print(f"📖 Loading new index: {new_index_path}")
    with open(new_index_path, "rb") as f:
        new_index = pickle.load(f)
    
    new_base_dir = Path(new_index["base_dir"])
    new_instrument = new_index["instrument"]
    new_shard_count = len(new_index["shards"])
    new_total_files = new_index["total_files"]
    
    print(f"   New shards: {new_shard_count}")
    print(f"   New files:  {new_total_files}")
    
    # 楽器チェック
    if existing_instrument != new_instrument:
        raise ValueError(
            f"Instrument mismatch: existing='{existing_instrument}' vs new='{new_instrument}'"
        )
    
    # 3. 新しいshardをコピー＆番号振り直し
    print()
    print("🔨 Copying and renumbering new shards...")
    
    added_shards = []
    
    for new_shard_info in new_index["shards"]:
        # 元のshard読み込み
        new_shard_path = new_base_dir / new_shard_info["path"]
        
        if not new_shard_path.exists():
            print(f"⚠️  Shard not found: {new_shard_path}, skipping")
            continue
        
        with open(new_shard_path, "rb") as f:
            shard_data = pickle.load(f)
        
        # 新しいshard番号
        new_shard_index = existing_shard_count + len(added_shards)
        shard_data["shard_index"] = new_shard_index
        
        # 保存先パス（既存ディレクトリに）
        new_shard_name = f"{existing_instrument}_metadata_v2_{new_shard_index:05d}.pkl"
        dst_shard_path = existing_base_dir / new_shard_name
        
        # アトミック書き込み
        tmp_path = existing_base_dir / f"{new_shard_name}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(dst_shard_path)
        
        # インデックスに追加
        added_shards.append({
            "path": new_shard_name,
            "count": shard_data["count"],
            "sha1": file_sha1(dst_shard_path),
        })
        
        print(f"   + Shard {new_shard_index}: {new_shard_name} ({shard_data['count']} files)")
    
    # 4. 既存インデックス更新
    print()
    print("📝 Updating index...")
    
    existing_index["shards"].extend(added_shards)
    existing_index["total_files"] += sum(s["count"] for s in added_shards)
    existing_index["fileset_hash"] = fileset_sha1([s["path"] for s in existing_index["shards"]])
    
    # 5. 保存（アトミック）
    tmp_index = existing_index_path.parent / f"{existing_index_path.name}.tmp"
    with open(tmp_index, "wb") as f:
        pickle.dump(existing_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_index.replace(existing_index_path)
    
    print()
    print("=" * 70)
    print("✅ Index updated successfully")
    print("=" * 70)
    print(f"Total shards: {len(existing_index['shards'])} (+{len(added_shards)})")
    print(f"Total files:  {existing_index['total_files']} (+{sum(s['count'] for s in added_shards)})")
    print(f"Index path:   {existing_index_path}")
    print("=" * 70)
    
    return existing_index_path


def main():
    parser = argparse.ArgumentParser(
        description="既存のLAMDAインデックスに新しいレコードを追加"
    )
    parser.add_argument(
        "--existing",
        type=Path,
        required=True,
        help="既存のインデックスpickleパス",
    )
    parser.add_argument(
        "--new",
        type=Path,
        required=True,
        help="新しいインデックスpickleパス",
    )
    
    args = parser.parse_args()
    
    # 存在チェック
    if not args.existing.exists():
        print(f"❌ Existing index not found: {args.existing}")
        return 1
    
    if not args.new.exists():
        print(f"❌ New index not found: {args.new}")
        return 1
    
    # 実行
    try:
        append_to_index(args.existing, args.new)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
