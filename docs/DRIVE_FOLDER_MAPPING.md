# Google Drive Folder Mapping Guide

## 📂 問題: フォルダ名の不一致

Google Drive内の実際のフォルダ名が、Stage2が期待する名前と異なっています。

### 実際のフォルダ (Google Drive):
```
output/
└── drumloops_cleaned/
```

### 期待されるフォルダ (Stage2):
```
output/
├── drum_metadata/
└── drum_cleaned/
```

---

## 🔧 解決策: 3つの方法

### 方法1: シンボリックリンク (推奨・最速)

```python
# Colabセル
!cd /content/composer4 && python scripts/adapt_drive_folders.py
```

**効果**:
- `drumloops_cleaned` → `drum_cleaned` にリンク
- `drumloops_metadata` → `drum_metadata` にリンク (存在する場合)
- 元のフォルダはそのまま、参照先だけ変更

---

### 方法2: 手動リンク作成

```bash
%%bash
cd /content/composer4/output

# drumloops_cleaned を drum_cleaned にリンク
ln -s drumloops_cleaned drum_cleaned

# drumloops_metadata があれば drum_metadata にリンク
if [ -d "drumloops_metadata" ]; then
    ln -s drumloops_metadata drum_metadata
fi

# 確認
ls -la
```

---

### 方法3: Google Drive内でリネーム (非推奨)

Google Drive側で直接リネーム:
- `drumloops_cleaned` → `drum_cleaned`
- `drumloops_metadata` → `drum_metadata`

**注意**: 他の場所からの参照が壊れる可能性あり

---

## 🚀 実行手順 (Colab)

### ステップ1: 最新コード取得

```bash
%%bash
cd /content/composer4
git pull origin main
```

### ステップ2: フォルダアダプタ実行

```python
!cd /content/composer4 && python scripts/adapt_drive_folders.py
```

**期待される出力**:
```
✅ drum_cleaned -> drumloops_cleaned (作成)
✅ drum_metadata -> drumloops_metadata (作成) or ⚠️ なし
```

### ステップ3: 再診断

```python
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
```

**期待される結果**:
- `drum_cleaned`: ✅ 存在
- `drum_metadata`: ✅ 存在 (drumloops_metadataがある場合) or ❌ 未作成

---

## 📋 診断結果に応じた次のステップ

### ケースA: drum_metadata も存在した

```
メタデータ (Stage1形式): ✅ OK
クリーンMIDI: ✅ OK
```

**→ Stage2直行**:
```bash
%%bash
cd /content/composer4
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

### ケースB: drum_metadata が存在しない

```
メタデータ (Stage1形式): ❌ NG
クリーンMIDI: ✅ OK
```

**→ メタデータだけ作成**:
```bash
%%bash
cd /content/composer4

# 1. メタデータ作成 (5-10分)
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata

# 2. Stage2実行 (10-30分)
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

---

## 🔍 Google Driveで確認すべきこと

### drumloops_cleaned の中身を確認

```python
!ls -la /content/composer4/output/drumloops_cleaned | head -n 20
```

**期待されるもの**:
- `cleaned/*.mid` (クリーンMIDIファイル)
- `cache/*.pkl` (キャッシュ、オプション)

または:
- `*.mid` (直下にMIDIファイル)

### drumloops_metadata があるか確認

```python
!ls -la /content/composer4/output/ | grep metadata
```

**もし存在すれば**:
```python
!ls -la /content/composer4/output/drumloops_metadata | head -n 20
```

**期待されるもの**:
- `*.pickle` ファイル (shard_0.pickle など)

---

## ⚠️ トラブルシューティング

### エラー: `cleaned/*.mid` が見つからない

**原因**: drumloops_cleaned の構造が違う

**確認**:
```python
!find /content/composer4/output/drumloops_cleaned -name "*.mid" | head -n 10
```

**MIDIファイルが別の場所にある場合**:
```bash
# 例: drumloops_cleaned/midi/*.mid の場合
ln -s drumloops_cleaned/midi /content/composer4/output/drum_cleaned/cleaned
```

### エラー: `FileNotFoundError: shard_0.pickle`

**原因**: drum_metadata が存在しないか、中身が空

**解決**: メタデータ作成を実行
```bash
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata
```

---

## 📊 完全な診断フロー (まとめ)

```python
# Colabで順番に実行

# 1. 最新版取得
!cd /content/composer4 && git pull origin main

# 2. フォルダ構造を確認
!ls -la /content/composer4/output/

# 3. フォルダアダプタ実行
!cd /content/composer4 && python scripts/adapt_drive_folders.py

# 4. 再診断
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py

# 5. 結果に応じて次のステップを実行
```

---

## 🎯 最短経路 (drumloops_cleaned が完全な場合)

```bash
%%bash
cd /content/composer4

# シンボリックリンク作成
ln -s output/drumloops_cleaned output/drum_cleaned

# メタデータ作成 (drumloops_metadataがない場合)
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata

# Stage2実行
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

**所要時間**: 15-40分 (メタデータ5-10分 + Stage2 10-30分)
