# Stage1 vs Stage2: 迅速判定ガイド

## 🎯 あなたの質問への回答

> stage1にもどってつくりなおすべきですか？
> クリーンmidiは生成されているようなんですが。

**答え**: **NO、完全なStage1は不要です！**

**推奨**: **メタデータだけ再作成** (5-10分) → Stage2実行 (10-30分)

---

## 🚀 Colabで今すぐ実行すべきこと

### ステップ1: 診断 (30秒)

```python
# Colabセル1: 最新版取得
!cd /content/composer4 && git pull origin main

# Colabセル2: 診断実行
!cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
```

### ステップ2: 診断結果を確認

診断スクリプトが以下を判定します:
- ✅ メタデータ (.pickle) が Stage1 マニフェスト形式か
- ✅ クリーンMIDI が存在するか
- ✅ 次に実行すべきコマンド

### ステップ3: 推奨コマンドを実行

**最も可能性の高いシナリオ** (クリーンMIDI済み):

```bash
%%bash
cd /content/composer4

# メタデータだけ再作成 (5-10分)
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata

# その後Stage2 (10-30分)
PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
  --metadata-index output/drum_metadata/shard_0.pickle \
  --metadata-dir output/drum_metadata \
  --input-dir output/drum_cleaned \
  --output-dir output/stage2_drum_iter1 \
  --config configs/lamda/drum_stage2.yaml \
  --print-summary
```

**合計時間**: 15-40分

---

## 📊 4つのシナリオ比較

| # | メタデータ | クリーンMIDI | アクション | 時間 |
|---|-----------|------------|----------|------|
| 1 | ✅ 新形式 | ✅ あり | Stage2直行 | 10-30分 |
| 2 | ❌ 古形式 | ✅ あり | メタデータ再作成 | 15-40分 ⭐ |
| 3 | ✅ 新形式 | ❌ なし | クリーニング再実行 | 30-70分 |
| 4 | ❌ 古形式 | ❌ なし | 完全Stage1 | 35-80分 |

**あなたのケース**: シナリオ2の可能性大 (15-40分で完了)

---

## ❓ よくある質問

### Q1: メタデータの「新形式」と「古形式」の違いは？

**古形式** (Stage2で使えない):
```python
{
    'file_001.mid': {
        'duration': 120.0,
        'tempo': 120
    }
}
```

**新形式** (Stage1マニフェスト、Stage2で使える):
```python
{
    'file_001.mid': {
        'midi_path': '/path/to/file_001.mid',  # ← これがある
        'loop_id': 'drum_001',                  # ← これがある
        'metadata': {...}                       # ← これがある
    }
}
```

### Q2: クリーンMIDIがあるのになぜメタデータ再作成が必要？

**理由**: メタデータとクリーンMIDIは独立したファイル

- **メタデータ** (.pickle): Stage2が「何を処理するか」のリスト
- **クリーンMIDI** (.mid): 実際のデータ

古いメタデータは新しいStage2スキーマと互換性がない可能性が高いです。

### Q3: 完全なStage1を再実行する必要があるのはいつ？

**必要な場合**:
- input/drum_raw が更新された（新しいMIDI追加）
- クリーンMIDIが実際には存在しない/壊れている
- 診断スクリプトが「Both NG」を報告

**不要な場合** (あなたのケース):
- クリーンMIDIは既にある
- メタデータだけ古い → **メタデータだけ再作成でOK**

### Q4: cache/*.pkl とは？

**役割**: Stage2の高速化キャッシュ (オプション)

- **ある場合**: Stage2が高速化される
- **ない場合**: MIDI直接パース（機能は同じ、少し遅い）

**重要**: cacheがなくてもStage2は動作します

---

## 🔧 トラブルシューティング

### エラー: `FileNotFoundError: shard_0.pickle`

**原因**: メタデータが存在しないか古い

**解決**:
```bash
python scripts/build_contract_records.py \
  --input-dir input/drum_raw \
  --output-dir output/drum_metadata
```

### エラー: `KeyError: 'midi_path'` または `KeyError: 'metadata'`

**原因**: メタデータが古い形式

**解決**: 上と同じ (メタデータ再作成)

### エラー: `FileNotFoundError: cleaned/*.mid`

**原因**: クリーンMIDIが実際には存在しない

**解決**:
```bash
python scripts/lamda_stage1_clean.py \
  --metadata-dir output/drum_metadata \
  --input-dir input/drum_raw \
  --output-dir output/drum_cleaned \
  --workers 8
```

---

## ✅ 成功の目印

### Stage2実行後、以下のファイルが生成される:

```
output/stage2_drum_iter1/
├── metrics_score.jsonl          # 各ファイルのスコア
├── stage2_summary.json           # 全体サマリー
├── velocity_coverage.json        # Velocity分布
└── canonical_events.parquet      # 正規化イベント
```

### 期待される出力例:

```
✅ Stage2 Complete!
   Processed: 150 files
   Pass rate: 92.3%
   Output: output/stage2_drum_iter1
```

---

## 📚 関連ドキュメント

| ドキュメント | 用途 |
|------------|------|
| `docs/STAGE1_VS_STAGE2_DECISION.md` | 詳細な判定フロー |
| `scripts/diagnose_stage1_stage2_colab.py` | Colab診断スクリプト |
| `docs/COLAB_QUICK_START.md` | Colabセットアップ全般 |

---

## 🎯 まとめ

### ✅ あなたがすべきこと (Colab)

1. **診断実行** (30秒):
   ```python
   !cd /content/composer4 && python scripts/diagnose_stage1_stage2_colab.py
   ```

2. **メタデータ再作成** (5-10分、必要な場合のみ):
   ```bash
   python scripts/build_contract_records.py \
     --input-dir input/drum_raw \
     --output-dir output/drum_metadata
   ```

3. **Stage2実行** (10-30分):
   ```bash
   PYTHONPATH=. python scripts/lamda_stage2_extractor.py \
     --metadata-index output/drum_metadata/shard_0.pickle \
     --metadata-dir output/drum_metadata \
     --input-dir output/drum_cleaned \
     --output-dir output/stage2_drum_iter1 \
     --config configs/lamda/drum_stage2.yaml \
     --print-summary
   ```

### ❌ 不要なこと

- ✅ **クリーンMIDI再作成は不要** (既にある)
- ✅ **完全なStage1は不要** (メタデータだけでOK)
- ✅ **20-40分の時間節約**

---

**結論**: クリーンMIDIがあるなら、メタデータだけ再作成して進めましょう！ 🚀
