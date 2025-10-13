# Colab Stage2 Setup Guide
# Google Drive共有フォルダからデータをダウンロードしてStage2を準備

## セル1: 基本セットアップ（リポジトリクローン＋依存関係）

```python
%%bash
# リポジトリクローン
if [ ! -d "/content/composer4" ]; then
    git clone https://github.com/kinoshitayoshihiro/composer4.git /content/composer4
    echo "✅ Repository cloned"
else
    cd /content/composer4
    git pull origin main
    echo "✅ Repository updated"
fi

# 依存関係インストール
cd /content/composer4
pip install -q torch transformers pytest gdown numpy scipy mido prettytable
echo "✅ Dependencies installed"
```

---

## セル2: Google Driveダウンロード（改良版）

```bash
%%bash
# Google Drive共有フォルダをダウンロード
FOLDER_URL="https://drive.google.com/drive/folders/1zUg85irbGgcHZggCGXiCwOZWHR6uKX5T?usp=sharing"
DEST="/content/composer4/_drive_download"
mkdir -p "$DEST"

echo "📥 Downloading from Google Drive..."
gdown --folder --fuzzy "$FOLDER_URL" -O "$DEST" || {
    echo "⚠️ First attempt failed, trying alternative..."
    gdown --folder "$FOLDER_URL" -O "$DEST" --remaining-ok
}

echo "✅ Download complete"
echo "Downloaded structure:"
ls -la "$DEST"
```

---

## セル3: データ構造診断（問題がある場合はここで確認）

```python
# 診断スクリプトを実行
!cd /content/composer4 && python scripts/diagnose_colab_setup.py
```

---

## セル4: 手動リンク作成（診断で問題が見つかった場合）

```bash
%%bash
# outputディレクトリを手動で探索
DEST="/content/composer4/_drive_download"

echo "🔎 Searching for output directory..."
find "$DEST" -type d -name "output" 2>/dev/null | while read output_path; do
    echo "Found: $output_path"
    echo "Contents:"
    ls -la "$output_path" | head -n 10
done

# 最初に見つかったoutputディレクトリをリンク
OUTPUT_SRC="$(find "$DEST" -type d -name "output" -print -quit)"
if [ -n "$OUTPUT_SRC" ]; then
    ln -sfn "$OUTPUT_SRC" /content/composer4/output
    echo "✅ Linked: $OUTPUT_SRC -> /content/composer4/output"
else
    echo "❌ No output directory found"
    echo ""
    echo "📂 Alternative: Manual setup options"
    echo "Option 1: Google Drive Mount"
    echo "  from google.colab import drive"
    echo "  drive.mount('/content/drive')"
    echo "  ln -s /content/drive/MyDrive/your_output_folder /content/composer4/output"
    echo ""
    echo "Option 2: Direct upload"
    echo "  Upload 'output' folder directly to /content/composer4/"
fi
```

---

## セル5: 別の方法 - Google Driveマウント（推奨）

```python
# Google Driveをマウントして直接アクセス
from google.colab import drive
drive.mount('/content/drive')

# ドライブ内のoutputフォルダをリンク
# ⚠️ 以下のパスを実際のGoogle Drive内のパスに変更してください
import os

# 例: Google Drive内の共有フォルダ構造を確認
print("📂 Google Drive structure:")
!ls -la /content/drive/MyDrive/

# outputフォルダを見つけてリンク
# パス例: /content/drive/MyDrive/composer4_data/output
drive_output_path = "/content/drive/MyDrive/composer4_data/output"  # ← 実際のパスに変更

if os.path.exists(drive_output_path):
    os.system(f"ln -sfn {drive_output_path} /content/composer4/output")
    print(f"✅ Linked: {drive_output_path} -> /content/composer4/output")
else:
    print(f"❌ Path not found: {drive_output_path}")
    print("💡 Google Drive内の正しいパスを確認してください")
```

---

## セル6: セットアップ検証

```python
import os
from pathlib import Path

print("🔍 Setup Verification")
print("=" * 60)

# 1. リポジトリ確認
repo = Path("/content/composer4")
print(f"\n📂 Repository: {repo}")
print(f"   Exists: {repo.exists()}")
if repo.exists():
    print(f"   Files: {len(list(repo.glob('*')))}")

# 2. outputリンク確認
output_link = repo / "output"
print(f"\n🔗 Output Link: {output_link}")
print(f"   Exists: {output_link.exists()}")
if output_link.exists():
    print(f"   Is symlink: {output_link.is_symlink()}")
    if output_link.is_symlink():
        target = output_link.resolve()
        print(f"   Target: {target}")
        print(f"   Target exists: {target.exists()}")

# 3. 必要なディレクトリ確認
if output_link.exists():
    print(f"\n📋 Required Directories:")
    required = ["drum_metadata", "drum_cleaned"]
    for dir_name in required:
        dir_path = output_link / dir_name
        exists = dir_path.exists()
        status = "✅" if exists else "⚠️"
        print(f"   {status} {dir_name}: {exists}")
        if exists:
            file_count = len(list(dir_path.glob('*')))
            print(f"      Files/Dirs: {file_count}")

# 4. GPU確認
print(f"\n🎮 GPU:")
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "No GPU"

print("\n" + "=" * 60)
if output_link.exists() and output_link.is_symlink():
    print("✅ Setup complete! Ready for Stage2")
    print("\n🚀 Next steps:")
    print("   1. cd /content/composer4")
    print("   2. bash scripts/run_stage2_drum.sh")
else:
    print("❌ Setup incomplete. Please check the errors above.")
```

---

## セル7: Stage2実行（セットアップ完了後）

```bash
%%bash
cd /content/composer4

# Stage2実行スクリプト（まだ作成していない場合は後で作成）
# bash scripts/run_stage2_drum.sh

# または直接実行
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

---

## トラブルシューティング

### 問題1: `gdown`でダウンロードできない

**原因**: Google Driveの共有設定、またはフォルダサイズ制限

**解決策**:
1. Google Driveマウント方式（セル5）を使用
2. 手動でZIPをダウンロード→Colabにアップロード

### 問題2: `output`ディレクトリが見つからない

**原因**: Google Drive内のフォルダ構造が想定と異なる

**解決策**:
```python
# ダウンロード内容を確認
!find /content/composer4/_drive_download -type d | head -n 30

# 手動でリンク作成
import os
actual_output_path = "/content/composer4/_drive_download/実際のパス/output"
os.system(f"ln -sfn {actual_output_path} /content/composer4/output")
```

### 問題3: メモリ不足

**原因**: Colabの無料版はRAM制限あり

**解決策**:
1. Colab Pro使用
2. バッチサイズを減らす
3. データを分割処理

---

## 推奨: Google Driveマウント方式（最も確実）

```python
from google.colab import drive
import os

# 1. マウント
drive.mount('/content/drive')

# 2. 共有フォルダへのパスを確認
# 「マイドライブ」→「共有アイテム」→「フォルダ名」の順に探す
print("Searching for shared folders...")
!ls -la "/content/drive/MyDrive/"
!ls -la "/content/drive/Shareddrives/" 2>/dev/null || echo "No shared drives"

# 3. 正しいパスを設定
# 例1: マイドライブ内
output_path = "/content/drive/MyDrive/composer4_data/output"

# 例2: 共有ドライブ内
# output_path = "/content/drive/Shareddrives/プロジェクト名/composer4_data/output"

# 4. リンク作成
if os.path.exists(output_path):
    os.system(f"ln -sfn {output_path} /content/composer4/output")
    print(f"✅ Success: {output_path}")
else:
    print(f"❌ Path not found. Please check:")
    print(f"   Expected: {output_path}")
    print("\n💡 Search for 'output' directory:")
    os.system('find /content/drive -type d -name "output" 2>/dev/null | head -n 5')
```
