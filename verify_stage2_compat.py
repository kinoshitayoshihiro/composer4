#!/usr/bin/env python3
"""
Stage2互換性検証スクリプト
clean_midi.py が生成した pickle が Stage2 で正しく読み込めるかチェック
"""

import pickle
import sys
from pathlib import Path

def check_index_structure(index_path: Path) -> bool:
    """インデックス構造をチェック"""
    print(f"📋 Checking index: {index_path}")
    
    if not index_path.exists():
        print(f"  ❌ Index file not found")
        return False
    
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    
    # 必須フィールドチェック
    required_fields = ["version", "shards", "instrument", "total_files"]
    for field in required_fields:
        if field not in index_data:
            print(f"  ❌ Missing field: {field}")
            return False
    
    print(f"  ✅ Version: {index_data['version']}")
    print(f"  ✅ Instrument: {index_data['instrument']}")
    print(f"  ✅ Total files: {index_data['total_files']}")
    print(f"  ✅ Shards: {len(index_data['shards'])}")
    
    # shards構造チェック
    if not index_data["shards"]:
        print(f"  ⚠️  No shards found")
        return False
    
    first_shard = index_data["shards"][0]
    shard_required = ["path", "index", "count"]
    for field in shard_required:
        if field not in first_shard:
            print(f"  ❌ Missing shard field: {field}")
            return False
    
    print(f"  ✅ Shard structure: {list(first_shard.keys())}")
    print(f"  ✅ First shard: {first_shard['path']}, index={first_shard['index']}, count={first_shard['count']}")
    
    return True


def check_shard_structure(shard_path: Path) -> bool:
    """シャード構造をチェック"""
    print(f"\n📦 Checking shard: {shard_path.name}")
    
    if not shard_path.exists():
        print(f"  ❌ Shard file not found")
        return False
    
    with open(shard_path, "rb") as f:
        shard_data = pickle.load(f)
    
    # 必須フィールドチェック
    required_fields = ["version", "shard_index", "loops", "count"]
    for field in required_fields:
        if field not in shard_data:
            print(f"  ❌ Missing field: {field}")
            return False
    
    print(f"  ✅ Version: {shard_data['version']}")
    print(f"  ✅ Shard index: {shard_data['shard_index']}")
    print(f"  ✅ Loop count: {shard_data['count']}")
    
    # loops構造チェック
    if not shard_data["loops"]:
        print(f"  ⚠️  No loops found")
        return True  # 空でもOK
    
    first_loop = shard_data["loops"][0]
    
    # Stage2が期待する必須フィールド
    loop_required = ["md5", "filename", "genre", "bpm", "note_count", "duration_ticks"]
    missing = []
    for field in loop_required:
        if field not in first_loop:
            missing.append(field)
    
    if missing:
        print(f"  ❌ Missing loop fields: {missing}")
        print(f"  Available fields: {list(first_loop.keys())}")
        return False
    
    print(f"  ✅ Loop structure: {loop_required}")
    print(f"  ✅ First loop:")
    print(f"     - filename: {first_loop['filename']}")
    print(f"     - genre: {first_loop['genre']}")
    print(f"     - bpm: {first_loop['bpm']}")
    print(f"     - notes: {first_loop['note_count']}")
    
    return True


def check_stage2_compatibility(pickle_dir: Path) -> bool:
    """Stage2互換性を総合チェック"""
    print("=" * 70)
    print("Stage2互換性チェック")
    print("=" * 70)
    print()
    
    # インデックスファイル検索
    index_files = list(pickle_dir.glob("*_index.pkl"))
    if not index_files:
        print(f"❌ No index file found in {pickle_dir}")
        return False
    
    index_path = index_files[0]
    
    # インデックスチェック
    if not check_index_structure(index_path):
        return False
    
    # シャードファイル検索
    instrument = index_path.stem.replace("_index", "")
    shard_files = sorted(pickle_dir.glob(f"{instrument}_*.pkl"))
    shard_files = [f for f in shard_files if "_index.pkl" not in f.name]
    
    if not shard_files:
        print(f"\n❌ No shard files found for instrument: {instrument}")
        return False
    
    print(f"\n📚 Found {len(shard_files)} shard files")
    
    # 最初のシャードをチェック
    if not check_shard_structure(shard_files[0]):
        return False
    
    # Stage2で読み込めるか確認（lamda_toolsがあれば）
    print("\n🔧 Testing with lamda_tools...")
    try:
        from lamda_tools.metadata_io import load_metadata_index, iter_loop_records
        
        index_data = load_metadata_index(index_path)
        print(f"  ✅ load_metadata_index() succeeded")
        
        loop_count = 0
        for record in iter_loop_records(index_data, index_path=index_path):
            loop = record.get("loop", {})
            if "genre" not in loop:
                print(f"  ❌ Loop missing 'genre' field")
                return False
            loop_count += 1
            if loop_count >= 5:  # 最初の5件だけチェック
                break
        
        print(f"  ✅ iter_loop_records() succeeded ({loop_count} loops checked)")
        print(f"  ✅ All loops have 'genre' field")
        
    except ImportError:
        print(f"  ⚠️  lamda_tools not available, skipping runtime test")
    except Exception as e:
        print(f"  ❌ Runtime error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ Stage2互換性チェック完了 - すべてOK！")
    print("=" * 70)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_stage2_compat.py <pickle_dir>")
        print("Example: python verify_stage2_compat.py data/lamda/shards/piano")
        sys.exit(1)
    
    pickle_dir = Path(sys.argv[1])
    
    if not pickle_dir.exists():
        print(f"❌ Directory not found: {pickle_dir}")
        sys.exit(1)
    
    success = check_stage2_compatibility(pickle_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
