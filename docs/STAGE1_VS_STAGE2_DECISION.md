# Stage1 vs Stage2: いつStage1に戻るべきか？

## 🎯 結論: クリーンMIDIがあれば**メタデータだけ**再作成でOK

あなたの状況:
- ✅ **クリーンMIDIは生成済み** (`output/drum_cleaned/*.mid`)
- ❓ **メタデータ(.pickle)の形式が不明**

---

## 📋 診断と判定フロー

### ステップ1: Colabで診断実行

```python
# Colabセル
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
```

**この診断が教えてくれること**:
1. メタデータがStage1マニフェスト形式か？ (新形式 vs 古形式)
2. クリーンMIDIが存在するか？
3. cacheファイル(.pkl)があるか？

---

### ステップ2: 診断結果に応じたアクション

#### ケース1: メタデータ✅ + クリーンMIDI✅ → **Stage2直行**

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

**所要時間**: 10-30分

---

#### ケース2: メタデータ❌ + クリーンMIDI✅ → **メタデータだけ再作成**

```bash
%%bash
cd /content/composer4

# メタデータだけ再作成 (5-10分)
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata

# その後Stage2
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

**所要時間**: 15-40分 (メタデータ5-10分 + Stage2 10-30分)

**💡 このケースが最も可能性高い**

---

#### ケース3: メタデータ✅ + クリーンMIDI❌ → **クリーニングだけ再実行**

```bash
%%bash
cd /content/composer4

# クリーニングだけ (20-40分)
python scripts/lamda_stage1_clean.py \
  --metadata-dir output/drum_metadata \
  --input-dir input/drum_raw \
  --output-dir output/drum_cleaned \
  --workers 8

# その後Stage2
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

**所要時間**: 30-70分 (クリーニング20-40分 + Stage2 10-30分)

---

#### ケース4: 両方❌ → **完全なStage1**

```bash
%%bash
cd /content/composer4

# 1. メタデータ作成 (5-10分)
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata

# 2. クリーニング (20-40分)
python scripts/lamda_stage1_clean.py \
  --metadata-dir output/drum_metadata \
  --input-dir input/drum_raw \
  --output-dir output/drum_cleaned \
  --workers 8

# 3. Stage2 (10-30分)
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

**所要時間**: 35-80分 (全工程)

---

## 🔍 メタデータ形式の見分け方 (手動確認)

### 古い形式 (Stage2で使えない)

```python
# 古いスキーマ例
{
    'file_001.mid': {
        'duration': 120.0,
        'tempo': 120,
        'notes': [...]
    }
}
```

**特徴**:
- `midi_path` キーがない
- `metadata` キーがない
- `loop_id` キーがない

### 新しい形式 (Stage1マニフェスト、Stage2で使える)

```python
# Stage1マニフェスト
{
    'file_001.mid': {
        'midi_path': '/path/to/file_001.mid',
        'loop_id': 'drum_001',
        'metadata': {
            'tempo': 120,
            'time_sig': '4/4',
            'duration_sec': 120.0
        }
    }
}
```

**特徴**:
- ✅ `midi_path` キーあり
- ✅ `metadata` キーあり
- ✅ `loop_id` キーあり

---

## 💡 推奨: 診断スクリプトを先に実行

```python
# Colabで実行
!cd /content/composer4 && git pull origin main  # 最新版取得
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
```

**診断結果が教えてくれること**:
1. 現在の状態（何が揃っているか）
2. 推奨アクション（何を実行すべきか）
3. 具体的なコマンド（コピペで使える）

---

## ⏱️ 時間見積もり

| シナリオ | メタデータ | クリーニング | Stage2 | 合計 |
|---------|-----------|------------|--------|------|
| **ケース1** | スキップ | スキップ | 10-30分 | **10-30分** |
| **ケース2** | 5-10分 | スキップ | 10-30分 | **15-40分** ⭐最短 |
| **ケース3** | スキップ | 20-40分 | 10-30分 | **30-70分** |
| **ケース4** | 5-10分 | 20-40分 | 10-30分 | **35-80分** |

**あなたの状況 (クリーンMIDI済み)**: ケース2の可能性大 → **15-40分で完了**

---

## 🎯 結論

### ✅ クリーンMIDIが既にあるなら

**メタデータだけ再作成すればOK** (5-10分)

```bash
%%bash
cd /content/composer4
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata
```

その後Stage2を実行 (10-30分)

**合計**: 15-40分

---

### ❌ 完全なStage1に戻る必要がある場合

- クリーンMIDIが実際には存在しない
- クリーンMIDIが壊れている
- input/drum_raw が更新された

この場合のみ完全なStage1 (35-80分)

---

## 🚀 次のアクション (Colabで実行)

### ステップ1: 診断
```python
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
```

### ステップ2: 診断結果に従ってコマンド実行

診断スクリプトが具体的なコマンドを表示してくれます。

---

## 📚 関連ドキュメント

- `docs/COLAB_QUICK_START.md` - Colabセットアップ
- `scripts/diagnose_stage1_stage2_colab.py` - 診断スクリプト
- `scripts/setup_colab_stage2.sh` - 自動セットアップ
