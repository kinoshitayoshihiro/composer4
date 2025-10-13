"""
Google Drive folder structure adapter for Stage1/Stage2
実際のフォルダ名と期待されるフォルダ名のマッピングを作成
"""

# Colabで実行してください

from pathlib import Path
import os

print("=" * 70)
print("🔗 Folder Structure Adapter")
print("=" * 70)

repo_root = Path("/content/composer4")
output_dir = repo_root / "output"

# 実際のフォルダを確認
print("\n📂 現在のoutput内容:")
if output_dir.exists():
    for item in output_dir.iterdir():
        print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
else:
    print("  ❌ outputディレクトリが存在しません")
    exit(1)

# シンボリックリンク作成
mappings = [
    ("drumloops_cleaned", "drum_cleaned"),
    ("drumloops_metadata", "drum_metadata"),
]

print("\n🔗 シンボリックリンク作成:")
for src_name, dst_name in mappings:
    src = output_dir / src_name
    dst = output_dir / dst_name
    
    if src.exists():
        if dst.exists():
            if dst.is_symlink():
                print(f"  ✅ {dst_name} -> {src_name} (既存)")
            else:
                print(f"  ⚠️ {dst_name} は既に存在 (シンボリックリンクではない)")
        else:
            os.symlink(src, dst)
            print(f"  ✅ {dst_name} -> {src_name} (作成)")
    else:
        print(f"  ⚠️ {src_name} が見つかりません")

# 検証
print("\n📊 検証:")
for _, dst_name in mappings:
    dst = output_dir / dst_name
    if dst.exists():
        if dst.is_symlink():
            target = dst.resolve()
            file_count = len(list(target.glob('*'))) if target.is_dir() else 0
            print(f"  ✅ {dst_name} -> {target.name} ({file_count} items)")
        else:
            print(f"  ℹ️ {dst_name} (実ディレクトリ)")
    else:
        print(f"  ❌ {dst_name} なし")

print("\n" + "=" * 70)
print("✅ フォルダマッピング完了")
print("=" * 70)
print("\n次のステップ:")
print("  python scripts/diagnose_stage1_stage2_colab.py")
