# Colab Quick Start for Stage2

## 問題: `output`ディレクトリが見つからない

**エラーメッセージ**:
```
⚠️ 共有フォルダ内に 'output' ディレクトリが見つかりません。フォルダ構成をご確認ください。
```

---

## 解決策: 3つの方法（推奨順）

### 🥇 方法1: Google Driveマウント（最も確実・推奨）

```python
# Colabセル1: Google Driveをマウント
from google.colab import drive
import os

drive.mount('/content/drive')

# セル2: 共有フォルダ内の'output'を探す
!find /content/drive -type d -name "output" 2>/dev/null | head -n 5

# セル3: 正しいパスをコピーしてリンク作成
# 上のfindコマンドで見つかったパスを使用
output_path = "/content/drive/MyDrive/実際のパス/output"  # ← 実際のパスに変更

os.system(f"ln -sfn {output_path} /content/composer4/output")
print(f"✅ Linked: {output_path}")

# セル4: 確認
!ls -la /content/composer4/output | head -n 20
```

---

### 🥈 方法2: 改良版セットアップスクリプト（自動検出）

```bash
%%bash
# Colabセル1: リポジトリ取得
cd /content
git clone https://github.com/kinoshitayoshihiro/composer4.git
cd composer4

# セル2: 改良版セットアップスクリプト実行
bash scripts/setup_colab_stage2.sh
```

**このスクリプトの特徴**:
- Google Driveから自動ダウンロード
- 複数のパスを自動探索
- 見つからない場合は詳細診断を表示
- シンボリックリンク自動作成

---

### 🥉 方法3: 診断ツール（問題特定）

```python
# Colabセル: 診断実行
!cd /content/composer4 && python scripts/diagnose_colab_setup.py
```

**診断内容**:
- ダウンロードしたフォルダ構造の表示
- `output`ディレクトリの場所を特定
- 必要なサブディレクトリの確認
- 推奨アクションの提示

---

## 手動フォールバック: 直接パス指定

```bash
%%bash
# ダウンロードした内容を確認
find /content/composer4/_drive_download -type d | head -n 30

# 'output'を手動で探す
OUTPUT_PATH="$(find /content/composer4/_drive_download -type d -name 'output' -print -quit)"
echo "Found: $OUTPUT_PATH"

# 手動リンク作成
if [ -n "$OUTPUT_PATH" ]; then
    ln -sfn "$OUTPUT_PATH" /content/composer4/output
    echo "✅ Link created"
    ls -la /content/composer4/output | head -n 20
else
    echo "❌ Not found. Check folder structure:"
    ls -R /content/composer4/_drive_download | head -n 50
fi
```

---

## トラブルシューティング

### ケース1: Google Driveの共有フォルダが見つからない

**原因**: 共有設定が正しくない、または共有フォルダIDが違う

**解決**:
1. Google Driveで共有フォルダを開く
2. URLから正しいフォルダIDをコピー
3. `setup_colab_stage2.sh`の`FOLDER_URL`を更新

### ケース2: `gdown`でダウンロードできない

**原因**: フォルダサイズが大きい、または権限問題

**解決**: 方法1（Driveマウント）を使用

### ケース3: ディレクトリ構造が違う

**原因**: Google Drive内のフォルダ構造が想定と異なる

**解決**:
```python
# 手動で構造を確認
!find /content/drive -name "*.mid" | head -n 10  # MIDIファイルを探す
!find /content/drive -name "*.pickle" | head -n 10  # メタデータを探す

# 見つかったディレクトリをリンク
import os
actual_path = "/content/drive/MyDrive/見つかったパス"
os.system(f"ln -sfn {actual_path} /content/composer4/output")
```

---

## 確認: セットアップ成功の目印

```python
import os
from pathlib import Path

# チェック1: outputリンクが存在
output = Path("/content/composer4/output")
print(f"Output link exists: {output.exists()}")

# チェック2: 必要なディレクトリが存在
required = ["drum_metadata", "drum_cleaned"]
for d in required:
    path = output / d
    print(f"{d}: {path.exists()}")

# チェック3: ファイル数確認
if output.exists():
    for d in required:
        path = output / d
        if path.exists():
            count = len(list(path.glob('*')))
            print(f"  {d}: {count} files")
```

**期待される出力**:
```
Output link exists: True
drum_metadata: True
drum_cleaned: True
  drum_metadata: 50+ files
  drum_cleaned: 100+ files
```

---

## 次のステップ（セットアップ成功後）

```bash
%%bash
cd /content/composer4

# Stage2実行
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

---

## 完全なColabノートブック例

詳細は `docs/COLAB_STAGE2_SETUP.md` を参照
