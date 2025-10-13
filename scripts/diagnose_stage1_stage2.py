#!/usr/bin/env python3
"""
Stage1/Stage2 状態診断スクリプト
既存のメタデータとクリーンMIDIの状態を確認し、Stage2実行可否を判定
"""

import sys
from pathlib import Path
import pickle
import json
from typing import Dict, List, Any

def check_pickle_schema(pickle_path: Path) -> Dict[str, Any]:
    """pickleファイルのスキーマを確認"""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # データ型とキー構造を分析
        data_type = type(data).__name__
        
        if isinstance(data, dict):
            keys = list(data.keys())
            sample_key = keys[0] if keys else None
            sample_value = data[sample_key] if sample_key else None
            
            return {
                "status": "dict",
                "num_entries": len(keys),
                "sample_key": str(sample_key),
                "sample_value_type": type(sample_value).__name__,
                "sample_value_keys": list(sample_value.keys()) if isinstance(sample_value, dict) else None,
                "is_stage1_manifest": _is_stage1_manifest(data)
            }
        elif isinstance(data, list):
            return {
                "status": "list",
                "num_entries": len(data),
                "sample_type": type(data[0]).__name__ if data else None,
                "is_stage1_manifest": False
            }
        else:
            return {
                "status": "unknown",
                "data_type": data_type,
                "is_stage1_manifest": False
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "is_stage1_manifest": False
        }

def _is_stage1_manifest(data: Dict) -> bool:
    """Stage1マニフェストの形式かチェック"""
    if not isinstance(data, dict):
        return False
    
    # Stage1マニフェストの期待されるキー
    expected_keys = {'midi_path', 'metadata', 'loop_id'}
    
    # サンプルエントリをチェック
    for value in list(data.values())[:3]:
        if isinstance(value, dict):
            if not expected_keys.issubset(value.keys()):
                return False
        else:
            return False
    
    return True

def check_cleaned_midi(cleaned_dir: Path) -> Dict[str, Any]:
    """クリーンMIDIディレクトリの状態確認"""
    if not cleaned_dir.exists():
        return {"status": "not_found", "count": 0}
    
    # cleaned/*.mid を探す
    midi_files = list(cleaned_dir.glob('cleaned/*.mid'))
    if not midi_files:
        # 直下も確認
        midi_files = list(cleaned_dir.glob('*.mid'))
    
    # cache/*.pkl を探す
    cache_files = list(cleaned_dir.glob('cache/*.pkl'))
    
    return {
        "status": "found",
        "midi_count": len(midi_files),
        "cache_count": len(cache_files),
        "has_cleaned_subdir": (cleaned_dir / "cleaned").exists(),
        "has_cache_subdir": (cleaned_dir / "cache").exists(),
        "sample_midi": str(midi_files[0]) if midi_files else None,
        "sample_cache": str(cache_files[0]) if cache_files else None
    }

def main():
    print("=" * 70)
    print("🔍 Stage1/Stage2 状態診断")
    print("=" * 70)
    
    # 基本パス設定
    repo_root = Path("/Volumes/SSD-SCTU3A/ラジオ用/music_21/composer2-3")
    if not repo_root.exists():
        repo_root = Path("/content/composer4")
    
    metadata_dir = repo_root / "output/drum_metadata"
    cleaned_dir = repo_root / "output/drum_cleaned"
    stage2_dir = repo_root / "output/stage2_drum_iter1"
    
    # ===== 1. メタデータ(.pickle)の確認 =====
    print("\n📂 1. メタデータディレクトリ: output/drum_metadata")
    
    if not metadata_dir.exists():
        print("  ❌ ディレクトリが存在しません")
        print("  💡 Stage1を実行してください: build_contract_records.py")
    else:
        pickle_files = list(metadata_dir.glob("*.pickle"))
        print(f"  ✅ ディレクトリ存在: {len(pickle_files)} pickleファイル")
        
        if not pickle_files:
            print("  ⚠️ .pickleファイルが見つかりません")
        else:
            # 最初のpickleファイルを詳細チェック
            sample_pickle = pickle_files[0]
            print(f"\n  🔍 サンプル分析: {sample_pickle.name}")
            schema = check_pickle_schema(sample_pickle)
            
            print(f"     Type: {schema.get('status')}")
            print(f"     Entries: {schema.get('num_entries', 'N/A')}")
            
            if schema.get('is_stage1_manifest'):
                print("     ✅ Stage1マニフェスト形式 (Stage2で使用可能)")
            else:
                print("     ⚠️ 古い形式の可能性 (Stage1を再実行推奨)")
                if schema.get('sample_value_keys'):
                    print(f"     Keys: {schema['sample_value_keys']}")
    
    # ===== 2. クリーンMIDIの確認 =====
    print("\n📂 2. クリーンMIDIディレクトリ: output/drum_cleaned")
    
    cleaned_status = check_cleaned_midi(cleaned_dir)
    
    if cleaned_status['status'] == 'not_found':
        print("  ❌ ディレクトリが存在しません")
        print("  💡 Stage1を実行してください: lamda_stage1_clean.py")
    else:
        print(f"  ✅ ディレクトリ存在")
        print(f"     MIDI files: {cleaned_status['midi_count']}")
        print(f"     Cache files: {cleaned_status['cache_count']}")
        
        if cleaned_status['has_cleaned_subdir']:
            print(f"     ✅ cleaned/ サブディレクトリあり")
        
        if cleaned_status['cache_count'] > 0:
            print(f"     ✅ cache/ サブディレクトリあり (高速化可能)")
        else:
            print(f"     ⚠️ cache なし (MIDI直接パース、少し遅い)")
        
        if cleaned_status['midi_count'] == 0:
            print("     ❌ MIDIファイルが見つかりません")
            print("     💡 Stage1を実行してください")
    
    # ===== 3. Stage2出力の確認 =====
    print("\n📂 3. Stage2出力ディレクトリ: output/stage2_drum_iter1")
    
    if stage2_dir.exists():
        print(f"  ✅ ディレクトリ存在（既に実行済みの可能性）")
        # 主要な出力ファイルをチェック
        expected_files = [
            "metrics_score.jsonl",
            "stage2_summary.json",
            "velocity_coverage.json",
            "canonical_events.parquet"
        ]
        
        for fname in expected_files:
            fpath = stage2_dir / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                print(f"     ✅ {fname}: {size_mb:.2f} MB")
            else:
                print(f"     ⚠️ {fname}: 未生成")
    else:
        print(f"  ℹ️ ディレクトリ未作成（Stage2未実行）")
    
    # ===== 4. 判定と推奨アクション =====
    print("\n" + "=" * 70)
    print("📋 判定と推奨アクション")
    print("=" * 70)
    
    # メタデータチェック
    metadata_ok = False
    if metadata_dir.exists():
        pickle_files = list(metadata_dir.glob("*.pickle"))
        if pickle_files:
            schema = check_pickle_schema(pickle_files[0])
            metadata_ok = schema.get('is_stage1_manifest', False)
    
    # クリーンMIDIチェック
    cleaned_ok = cleaned_status['status'] == 'found' and cleaned_status['midi_count'] > 0
    
    if metadata_ok and cleaned_ok:
        print("\n✅ Stage2実行可能!")
        print("\n🚀 次のコマンドを実行してください:")
        print("\n```bash")
        print("PYTHONPATH=. python scripts/lamda_stage2_extractor.py \\")
        print("  --metadata-index output/drum_metadata/shard_0.pickle \\")
        print("  --metadata-dir output/drum_metadata \\")
        print("  --input-dir output/drum_cleaned \\")
        print("  --output-dir output/stage2_drum_iter1 \\")
        print("  --config configs/lamda/drum_stage2.yaml \\")
        print("  --print-summary")
        print("```")
        
        if cleaned_status['cache_count'] > 0:
            print("\n💡 cache/*.pkl が存在するため高速化されます")
        else:
            print("\n💡 cache なし: MIDIを直接パースします（機能は同じ、少し遅い）")
    
    elif not metadata_ok and cleaned_ok:
        print("\n⚠️ メタデータが古い形式です")
        print("\n💡 Stage1 (メタデータ再作成のみ) を実行してください:")
        print("\n```bash")
        print("# 1. メタデータ再作成")
        print("python scripts/build_contract_records.py \\")
        print("  --input-dir input/drum_raw \\")
        print("  --output-dir output/drum_metadata")
        print("\n# 2. クリーンMIDIは既存のものを使用")
        print("# (lamda_stage1_clean.py はスキップ可能)")
        print("\n# 3. Stage2実行")
        print("PYTHONPATH=. python scripts/lamda_stage2_extractor.py \\")
        print("  --metadata-index output/drum_metadata/shard_0.pickle \\")
        print("  --metadata-dir output/drum_metadata \\")
        print("  --input-dir output/drum_cleaned \\")
        print("  --output-dir output/stage2_drum_iter1 \\")
        print("  --config configs/lamda/drum_stage2.yaml \\")
        print("  --print-summary")
        print("```")
    
    elif metadata_ok and not cleaned_ok:
        print("\n⚠️ クリーンMIDIが不足しています")
        print("\n💡 Stage1 (クリーニングのみ) を実行してください:")
        print("\n```bash")
        print("python scripts/lamda_stage1_clean.py \\")
        print("  --metadata-dir output/drum_metadata \\")
        print("  --input-dir input/drum_raw \\")
        print("  --output-dir output/drum_cleaned \\")
        print("  --workers 8")
        print("```")
    
    else:
        print("\n❌ Stage1から実行が必要です")
        print("\n💡 完全なStage1実行:")
        print("\n```bash")
        print("# 1. メタデータ作成")
        print("python scripts/build_contract_records.py \\")
        print("  --input-dir input/drum_raw \\")
        print("  --output-dir output/drum_metadata")
        print("\n# 2. クリーニング")
        print("python scripts/lamda_stage1_clean.py \\")
        print("  --metadata-dir output/drum_metadata \\")
        print("  --input-dir input/drum_raw \\")
        print("  --output-dir output/drum_cleaned \\")
        print("  --workers 8")
        print("\n# 3. Stage2実行")
        print("PYTHONPATH=. python scripts/lamda_stage2_extractor.py \\")
        print("  --metadata-index output/drum_metadata/shard_0.pickle \\")
        print("  --metadata-dir output/drum_metadata \\")
        print("  --input-dir output/drum_cleaned \\")
        print("  --output-dir output/stage2_drum_iter1 \\")
        print("  --config configs/lamda/drum_stage2.yaml \\")
        print("  --print-summary")
        print("```")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
