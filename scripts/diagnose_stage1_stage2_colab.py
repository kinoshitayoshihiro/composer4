"""
Colab専用: Stage1/Stage2 状態診断スクリプト
Google Drive内のメタデータとクリーンMIDIの状態を確認
"""

# このセルをColabで実行してください

from pathlib import Path
import pickle

def check_pickle_schema(pickle_path: Path):
    """pickleファイルの形式確認"""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            keys = list(data.keys())
            if not keys:
                return {"status": "empty_dict", "is_stage1": False}
            
            sample_key = keys[0]
            sample_value = data[sample_key]
            
            # Stage1マニフェストの特徴: midi_path, metadata, loop_id を持つ
            is_stage1 = False
            if isinstance(sample_value, dict):
                has_midi_path = 'midi_path' in sample_value
                has_metadata = 'metadata' in sample_value
                has_loop_id = 'loop_id' in sample_value
                is_stage1 = has_midi_path and has_metadata and has_loop_id
            
            return {
                "status": "dict",
                "entries": len(keys),
                "sample_keys": list(sample_value.keys()) if isinstance(sample_value, dict) else None,
                "is_stage1": is_stage1
            }
        elif isinstance(data, list):
            return {
                "status": "list",
                "entries": len(data),
                "is_stage1": False
            }
        else:
            return {
                "status": f"unknown ({type(data).__name__})",
                "is_stage1": False
            }
    except Exception as e:
        return {"status": "error", "error": str(e), "is_stage1": False}

# ===== 診断開始 =====
print("=" * 70)
print("🔍 Colab Stage1/Stage2 状態診断")
print("=" * 70)

repo_root = Path("/content/composer4")
output_dir = repo_root / "output"

# 1. outputディレクトリの確認
print(f"\n📂 1. Output Directory: {output_dir}")
if not output_dir.exists():
    print("  ❌ output ディレクトリが見つかりません")
    print("\n  💡 以下のいずれかを実行してください:")
    print("     1. Google Drive マウント + リンク作成")
    print("     2. bash scripts/setup_colab_stage2.sh")
    print("     3. python scripts/diagnose_colab_setup.py")
    print("\n  詳細: docs/COLAB_QUICK_START.md")
    exit(1)
else:
    print(f"  ✅ 存在します")
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    print(f"     サブディレクトリ: {len(subdirs)}")
    for d in subdirs:
        print(f"       - {d.name}")

# 2. drum_metadata の確認
print(f"\n📂 2. Metadata Directory: output/drum_metadata")
metadata_dir = output_dir / "drum_metadata"

if not metadata_dir.exists():
    print("  ❌ drum_metadata が存在しません")
    print("  💡 Stage1を実行してください:")
    print("     python scripts/build_contract_records.py \\")
    print("       --input-dir input/drum_raw \\")
    print("       --output-dir output/drum_metadata")
else:
    pickle_files = list(metadata_dir.glob("*.pickle"))
    print(f"  ✅ 存在: {len(pickle_files)} pickleファイル")
    
    if pickle_files:
        # 最初のpickleを分析
        sample = pickle_files[0]
        print(f"\n  🔍 {sample.name} を分析:")
        schema = check_pickle_schema(sample)
        print(f"     Type: {schema['status']}")
        print(f"     Entries: {schema.get('entries', 'N/A')}")
        
        if schema['is_stage1']:
            print("     ✅ Stage1マニフェスト形式 (Stage2で使用可能)")
        else:
            print("     ⚠️ 古い形式の可能性あり")
            if schema.get('sample_keys'):
                print(f"     Keys: {schema['sample_keys']}")
            print("\n     💡 Stage1を再実行推奨:")
            print("        python scripts/build_contract_records.py \\")
            print("          --input-dir input/drum_raw \\")
            print("          --output-dir output/drum_metadata")

# 3. drum_cleaned の確認
print(f"\n📂 3. Cleaned MIDI Directory: output/drum_cleaned")
cleaned_dir = output_dir / "drum_cleaned"

if not cleaned_dir.exists():
    print("  ❌ drum_cleaned が存在しません")
    print("  💡 Stage1を実行してください:")
    print("     python scripts/lamda_stage1_clean.py \\")
    print("       --metadata-dir output/drum_metadata \\")
    print("       --input-dir input/drum_raw \\")
    print("       --output-dir output/drum_cleaned \\")
    print("       --workers 8")
else:
    # cleaned/*.mid を探す
    midi_files = list(cleaned_dir.glob("cleaned/*.mid"))
    if not midi_files:
        midi_files = list(cleaned_dir.glob("*.mid"))
    
    # cache/*.pkl を探す
    cache_files = list(cleaned_dir.glob("cache/*.pkl"))
    
    print(f"  ✅ 存在")
    print(f"     MIDI files: {len(midi_files)}")
    print(f"     Cache files: {len(cache_files)}")
    
    if (cleaned_dir / "cleaned").exists():
        print(f"     ✅ cleaned/ サブディレクトリあり")
    
    if len(cache_files) > 0:
        print(f"     ✅ cache/ あり (高速化)")
    else:
        print(f"     ℹ️ cache なし (MIDI直接パース)")
    
    if len(midi_files) == 0:
        print("     ❌ MIDIファイルが見つかりません!")
        print("     💡 Stage1 (クリーニング) を実行してください")

# 4. stage2出力の確認
print(f"\n📂 4. Stage2 Output: output/stage2_drum_iter1")
stage2_dir = output_dir / "stage2_drum_iter1"

if stage2_dir.exists():
    print(f"  ✅ 存在 (既に実行済み)")
    
    expected = ["metrics_score.jsonl", "stage2_summary.json", 
                "velocity_coverage.json", "canonical_events.parquet"]
    
    for fname in expected:
        fpath = stage2_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"     ✅ {fname}: {size_mb:.2f} MB")
        else:
            print(f"     ⚠️ {fname}: 未生成")
else:
    print(f"  ℹ️ 未作成 (Stage2未実行)")

# ===== 判定 =====
print("\n" + "=" * 70)
print("📋 判定と推奨アクション")
print("=" * 70)

metadata_ok = False
if metadata_dir.exists():
    pickles = list(metadata_dir.glob("*.pickle"))
    if pickles:
        schema = check_pickle_schema(pickles[0])
        metadata_ok = schema.get('is_stage1', False)

cleaned_ok = False
if cleaned_dir.exists():
    midis = list(cleaned_dir.glob("cleaned/*.mid")) or list(cleaned_dir.glob("*.mid"))
    cleaned_ok = len(midis) > 0

print(f"\nメタデータ (Stage1形式): {'✅ OK' if metadata_ok else '❌ NG'}")
print(f"クリーンMIDI: {'✅ OK' if cleaned_ok else '❌ NG'}")

if metadata_ok and cleaned_ok:
    print("\n✅ Stage2実行可能!")
    print("\n🚀 次のコマンド:")
    print("\n%%bash")
    print("cd /content/composer4")
    print("PYTHONPATH=. python scripts/lamda_stage2_extractor.py \\")
    print("  --metadata-index output/drum_metadata/shard_0.pickle \\")
    print("  --metadata-dir output/drum_metadata \\")
    print("  --input-dir output/drum_cleaned \\")
    print("  --output-dir output/stage2_drum_iter1 \\")
    print("  --config configs/lamda/drum_stage2.yaml \\")
    print("  --print-summary")
    
elif not metadata_ok and cleaned_ok:
    print("\n⚠️ メタデータが古い形式")
    print("\n💡 メタデータだけ再作成:")
    print("\n%%bash")
    print("cd /content/composer4")
    print("python scripts/build_contract_records.py \\")
    print("  --input-dir input/drum_raw \\")
    print("  --output-dir output/drum_metadata")
    print("\n# その後Stage2を実行")
    
elif metadata_ok and not cleaned_ok:
    print("\n⚠️ クリーンMIDIが不足")
    print("\n💡 クリーニングだけ実行:")
    print("\n%%bash")
    print("cd /content/composer4")
    print("python scripts/lamda_stage1_clean.py \\")
    print("  --metadata-dir output/drum_metadata \\")
    print("  --input-dir input/drum_raw \\")
    print("  --output-dir output/drum_cleaned \\")
    print("  --workers 8")
    
else:
    print("\n❌ Stage1から実行が必要")
    print("\n💡 完全なStage1:")
    print("\n%%bash")
    print("cd /content/composer4")
    print("\n# 1. メタデータ作成")
    print("python scripts/build_contract_records.py \\")
    print("  --input-dir input/drum_raw \\")
    print("  --output-dir output/drum_metadata")
    print("\n# 2. クリーニング")
    print("python scripts/lamda_stage1_clean.py \\")
    print("  --metadata-dir output/drum_metadata \\")
    print("  --input-dir input/drum_raw \\")
    print("  --output-dir output/drum_cleaned \\")
    print("  --workers 8")

print("\n" + "=" * 70)
