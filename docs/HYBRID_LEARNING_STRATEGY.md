# ハイブリッド学習データ戦略：実装計画

## 📋 概要

ChatGPTとの議論から得られた「実データ中心＋合成で穴埋め」戦略の実装計画

**基本方針:**
- 実データ: 合成データ = 80:20 からスタート
- 楽器別に比率調整
- 奏法分布の偏りを定量化→補完

---

## 🎯 Phase 1: 分布可視化・定量化（即座実施）

### Task 1.1: 奏法分布分析スクリプト ✅

**実装済み:** `scripts/analyze_technique_distribution.py`

**機能:**
- Guitar: arpeggio/strum/chord_block/mixed の比率
- Bass: on_grid/slight_swing/syncopated の比率  
- Strings: legato/staccato/mixed の比率
- Piano: dynamics_range/rhythm_diversity の統計

**実行例:**
```bash
# Guitar分析
python scripts/analyze_technique_distribution.py \
  --instrument guitar \
  --input output/test_results/guitar_full.json \
  --output reports/guitar_technique_dist.json

# Bass分析
python scripts/analyze_technique_distribution.py \
  --instrument bass \
  --input output/test_results/bass_full.json \
  --output reports/bass_groove_dist.json

# Strings分析（完了後）
python scripts/analyze_technique_distribution.py \
  --instrument strings \
  --input output/test_results/strings_full.json \
  --output reports/strings_articulation_dist.json

# Piano分析
python scripts/analyze_technique_distribution.py \
  --instrument piano \
  --input output/test_results/piano_melody_full.json \
  --output reports/piano_expression_dist.json
```

**期待される発見:**
- Guitar: arpeggio > 50% の確認
- Bass: on_grid > 70% の確認
- Strings: staccato > 40% または legato < 50% の確認
- Piano: 高品質確認

---

### Task 1.2: 分布レポート生成

**TODO:** `scripts/generate_distribution_report.py`

**機能:**
- 全楽器の分布を統合
- 不足領域の自動検出
- 合成データ発注量の見積もり

**出力例:**
```json
{
  "guitar": {
    "current": {"arpeggio": 0.65, "strum": 0.20, "mixed": 0.15},
    "target": {"arpeggio": 0.40, "strum": 0.40, "mixed": 0.20},
    "deficit": {"strum": 0.20},
    "synthetic_needed": {
      "strum_downup_8th": 150,
      "strum_mixed_16th": 100,
      "fingerpicking": 50
    }
  },
  "bass": {
    "current": {"on_grid": 0.75, "slight_swing": 0.20, "syncopated": 0.05},
    "target": {"on_grid": 0.60, "slight_swing": 0.30, "syncopated": 0.10},
    "deficit": {"slight_swing": 0.10, "syncopated": 0.05},
    "synthetic_needed": {
      "walking_bass": 60,
      "chromatic_passing": 30
    }
  }
}
```

---

## 🏗️ Phase 2: データ再バランス（短期）

### Task 2.1: 実データ再サンプリング

**TODO:** `scripts/rebalance_training_data.py`

**機能:**
- 奏法/テンポ/スタイル別に層化サンプリング
- 不足領域はオーバーサンプリング
- 過多領域はアンダーサンプリング

**実行例:**
```bash
python scripts/rebalance_training_data.py \
  --instrument guitar \
  --input output/slakh/clean/guitar \
  --distribution reports/guitar_technique_dist.json \
  --output output/slakh/balanced/guitar \
  --strategy oversample  # or undersample or smote
```

**戦略オプション:**
- `oversample`: 不足領域を複製（データ拡張あり）
- `undersample`: 過多領域を削減
- `smote`: Synthetic Minority Over-sampling（MIDI向けカスタム）

---

### Task 2.2: データ拡張（Augmentation）

**TODO:** `scripts/augment_midi_variations.py`

**機能:**
- Bass: ±10-20ms ランダムIOI揺らぎ
- Guitar: ベロシティ微調整、音価微調整
- Strings: レガート接続強化
- Piano: 既存で十分（POP909高品質）

**実行例:**
```bash
# Bass ヒューマナイズ
python scripts/augment_midi_variations.py \
  --instrument bass \
  --input output/slakh/balanced/bass \
  --output output/slakh/augmented/bass \
  --timing-jitter 15  # ±15ms
  --velocity-range 5   # ±5

# Guitar ベロシティ多様化
python scripts/augment_midi_variations.py \
  --instrument guitar \
  --input output/slakh/balanced/guitar \
  --output output/slakh/augmented/guitar \
  --velocity-curve humanize \
  --strum-variation on
```

---

## 🎨 Phase 3: 合成データ生成（中期）

### Task 3.1: 合成データ発注マニフェスト

**TODO:** `configs/synthetic_data_manifest.yaml`

**構造:**
```yaml
guitar:
  strum_patterns:
    - technique: strum_downup_8th
      count: 150
      tempo_range: [80, 140]
      styles: [pop, rock, folk]
      
    - technique: strum_mixed_16th
      count: 100
      tempo_range: [100, 160]
      styles: [rock, punk, metal]
      
  fingerpicking:
    - technique: travis_picking
      count: 50
      tempo_range: [60, 100]
      styles: [folk, country]

bass:
  walking_bass:
    - count: 60
      tempo_range: [100, 140]
      styles: [jazz, blues]
      
  chromatic_passing:
    - count: 30
      tempo_range: [80, 120]
      styles: [jazz, funk]

strings:
  legato_enhancement:
    - technique: sustained_legato
      count: 100
      duration_range: [1.0, 4.0]
      styles: [classical, cinematic]
      
  spiccato:
    - count: 30
      duration_range: [0.2, 0.5]
      styles: [classical, contemporary]
```

---

### Task 3.2: Suno→MIDI パイプライン

**既存:** `emotion_humanizer.py` を拡張

**追加機能:**
- マニフェスト駆動生成
- 自動品質ゲート（Stage2メトリクス）
- タグ付け（source=synthetic）

**実行例:**
```bash
# マニフェスト駆動生成
python scripts/generate_synthetic_from_manifest.py \
  --manifest configs/synthetic_data_manifest.yaml \
  --output output/synthetic/guitar \
  --quality-gate configs/lamda/guitar_stage2.yaml \
  --threshold 45.0  # 実データより厳しく
```

---

### Task 3.3: 合成データ品質保証

**TODO:** `scripts/validate_synthetic_quality.py`

**機能:**
- Stage2メトリクス自動評価
- 実データより厳しい閾値（+5〜10%）
- 人間レビューキュー（100件/batch）

**実行例:**
```bash
python scripts/validate_synthetic_quality.py \
  --input output/synthetic/guitar \
  --config configs/lamda/guitar_stage2.yaml \
  --threshold 50.0  # 実データ40.0より厳しく \
  --review-queue output/synthetic/review_queue.json
```

---

## 🔀 Phase 4: ハイブリッド統合（中期）

### Task 4.1: データマージ

**TODO:** `scripts/merge_real_synthetic.py`

**機能:**
- 実データ + 合成データのブレンド
- 比率管理（80:20 デフォルト）
- メタデータタグ付け

**実行例:**
```bash
python scripts/merge_real_synthetic.py \
  --real output/slakh/augmented/guitar \
  --synthetic output/synthetic/guitar \
  --output output/training/guitar \
  --ratio 80:20 \
  --tag-source on
```

**出力メタデータ例:**
```json
{
  "file": "Track00123_guitar.mid",
  "source": "real",
  "dataset": "slakh",
  "technique": "arpeggio",
  "quality_score": 0.65
}

{
  "file": "synthetic_strum_001.mid",
  "source": "synthetic",
  "generator": "emotion_humanizer_v1",
  "technique": "strum_downup_16th",
  "quality_score": 0.72
}
```

---

### Task 4.2: Train/Val/Test 分割

**TODO:** `scripts/prepare_stratified_splits.py`

**機能:**
- 奏法/テンポ/スタイル別に層化分割
- 合成データは Train のみ
- Val/Test は実データのみ（リーク防止）

**実行例:**
```bash
python scripts/prepare_stratified_splits.py \
  --input output/training/guitar \
  --output output/splits/guitar \
  --strata technique,tempo,style \
  --ratio 80:10:10 \
  --synthetic-only-train on
```

**出力:**
```
output/splits/guitar/
├── train/          # 実80% + 合成20%
├── validation/     # 実データのみ 10%
└── test/           # 実データのみ 10%
```

---

## 📊 Phase 5: KPI監視・A/B評価（長期）

### Task 5.1: 内部KPIダッシュボード

**TODO:** `scripts/monitor_kpis.py`

**監視指標:**
- 楽器別 Stage2 通過率（奏法別）
- 合成データ比率
- 品質スコア分布
- 奏法バランス指数

**実行例:**
```bash
python scripts/monitor_kpis.py \
  --training-dir output/training \
  --output reports/kpi_dashboard.html
```

---

### Task 5.2: 外部ベンチマーク

**TODO:** `scripts/evaluate_external_benchmark.py`

**ベンチマーク:**
- MAESTRO (Piano)
- GuitarSet (Guitar)
- ASAP (Strings)
- Groove MIDI Dataset (Drums)

**実行例:**
```bash
python scripts/evaluate_external_benchmark.py \
  --model output/models/guitar_generator_v1.ckpt \
  --benchmark data/external/guitarset \
  --metrics pitch_accuracy,rhythm_consistency,style_coherence
```

---

### Task 5.3: A/B比較（合成あり vs なし）

**TODO:** `scripts/compare_ablation.py`

**比較項目:**
- 実データのみ（ベースライン）
- 実:合成 = 90:10
- 実:合成 = 80:20
- 実:合成 = 70:30

**実行例:**
```bash
python scripts/compare_ablation.py \
  --baseline output/models/guitar_real_only \
  --variant1 output/models/guitar_90_10 \
  --variant2 output/models/guitar_80_20 \
  --variant3 output/models/guitar_70_30 \
  --metrics external_benchmark,internal_gate,subjective_rating
```

---

## 🚨 リスク管理

### セルフトレーニング崩壊の回避

**対策:**
1. 合成データの種は実データから直接コピー禁止
2. プロンプト/和声/リズムは多系統（複数ルール）
3. 合成比率上限: 30%（段階的拡大、A/Bで監視）

### 外部検証セットのリーク防止

**対策:**
1. Val/Test は実データのみ
2. 外部ベンチマーク（MAESTRO/GuitarSet等）は学習に使わない
3. メタデータで source タグ必須

### 品質劣化の早期検出

**対策:**
1. Nightly KPI監視
2. 外部ベンチマーク自動評価
3. 合成データの品質ゲート（実データ+5〜10%）

---

## 📅 タイムライン

### 即座（今週）
- ✅ Task 1.1: 奏法分布分析スクリプト実装
- ⏳ Task 1.2: 分布レポート生成
- ⏳ Guitar/Bass/Piano 分布分析実行
- ⏳ Strings完了待ち→分析実行

### 短期（1-2週間）
- Task 2.1: 実データ再サンプリング
- Task 2.2: データ拡張（Augmentation）
- Task 3.1: 合成データマニフェスト作成

### 中期（1ヶ月）
- Task 3.2: Suno→MIDI パイプライン構築
- Task 3.3: 合成データ品質保証
- Task 4.1: データマージ
- Task 4.2: 層化分割

### 長期（2-3ヶ月）
- Task 5.1: KPIダッシュボード
- Task 5.2: 外部ベンチマーク評価
- Task 5.3: A/B比較実験

---

## 🤔 議論ポイント

### 1. 合成データの初期比率
- ChatGPT提案: 80:20
- 私の提案: 楽器別調整
  - Guitar: 70:30 (ストラム不足)
  - Bass: 85:15 (グリッド優先)
  - Strings: 70:30 (legato不足)
  - Piano: 90:10 (高品質)

**質問:** この比率でスタートしますか？

### 2. Suno→MIDI の品質
- 現状: emotion_humanizer.py が存在
- 懸念: MIDIへの変換精度は？

**質問:** 既存のSuno→MIDI変換の品質レビューは必要？

### 3. 外部ベンチマークの優先度
- MAESTRO (Piano): 容易
- GuitarSet: やや難
- ASAP (Strings): 中程度

**質問:** どの楽器から外部検証を始めますか？

### 4. 実データ再バランス vs 合成
- 再バランス: 既存データの層化サンプリング
- 合成: 新規データ生成

**質問:** どちらを優先？それとも並行？

---

## 📝 次のアクション

1. **Strings実行完了を待つ**
2. **全楽器の分布分析を実行**
3. **分布レポートを確認→議論**
4. **Task 2.1（再サンプリング）or Task 3.1（合成マニフェスト）を決定**

---

## 💬 私の総評

ChatGPTの提案は非常に戦略的で実践的です。特に:

✅ **賛成:**
- ハイブリッド戦略（実:合成 = 80:20）
- タグ付け必須
- 外部ベンチマーク
- セルフトレーニング回避策

⚠️ **懸念:**
- Suno→MIDI品質の担保
- 合成データの品質ゲート強化
- 分布の定量化が先決

💡 **追加提案:**
- まず分布を見てから合成量を決める
- 実データ再バランスを優先（即効性）
- 合成は不足が確定した領域のみ
