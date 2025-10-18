# Manifest-Driven Hybrid Data Architecture - Implementation Report

**実装日**: 2025年10月17日  
**統合**: ChatGPT提案アーキテクチャ

---

## 📊 実装サマリー

### ✅ 完了項目

1. **Technique Distribution Analysis** (統合分布カウント生成)
2. **Gap Estimation** (不足量推定)
3. **Manifest Generation** (manifest駆動ジョブ生成)
4. **Targets Definition** (targets_hybrid.yaml)

### システム構成

```
scripts/
├── generate_distribution_counts.py  # Stage2結果→統合分布JSON
├── estimate_gaps_and_emit_manifest.py  # Gap→manifest JSONL
└── run_manifest.py  # (次フェーズ) manifest実行→shard pickle

configs/
└── targets_hybrid.yaml  # 楽器×テンポ×奏法目標

reports/
└── integrated_distribution_counts.json  # 現状分布（実カウント）

manifests/
└── manifest_20251017.jsonl  # 7,888ジョブ
```

---

## 📈 現状分布分析 (3,282 real files)

### Guitar (1,422 files)
**Mid-tempo分布:**
- chord_block: 419
- strum: 266 ⚠️
- mixed: 704
- arpeggio: 33 ⚠️

**課題:**
- Strum極端に不足 (目標1,200 vs 実266 = **934不足**)
- Arpeggio極端に不足 (目標600 vs 実33 = **567不足**)

### Bass (584 files)
**Mid-tempo分布:**
- walking: 148
- slight_swing: 423
- on_grid: 13

**課題:**
- Grid-locked patterns (100%合格 = 識別力不足)
- Walking bass不足 (目標300 vs 実148 = **152不足**)

### Strings (999 files)
**Slow-tempo分布:**
- staccato: 182
- mixed: 734
- legato: 83 ⚠️⚠️⚠️

**課題:**
- **Legato極端に不足** (目標840 vs 実83 = **757不足 - 最優先ギャップ!**)
- Spiccato完全欠如 (目標280 vs 実0 = **280不足**)

### Piano (277 files)
**Mid-tempo分布:**
- standard: 127
- expressive: 150

**状況:**
- POP909高品質データ
- 最小限の合成で十分 (90:10戦略)

---

## 🎯 Manifest生成結果

### 総合成データ必要数: **7,888 files**

**楽器別内訳:**
- Guitar: 2,901 files
- Strings: 2,135 files
- Bass: 1,652 files
- Piano: 1,200 files

**テンポ帯別:**
- Slow: 2,635 files
- Mid: 3,553 files
- Fast: 1,700 files

### TOP 10 Priority Gaps

| 順位 | 奏法 | 不足数 | 優先度 |
|-----|------|-------|-------|
| 1 | **guitar_strum** | **1,554** | 🔴 CRITICAL |
| 2 | **strings_legato** | **1,117** | 🔴 CRITICAL |
| 3 | guitar_arpeggio | 1,007 | 🟠 HIGH |
| 4 | bass_pick | 900 | 🟠 HIGH |
| 5 | strings_spiccato | 600 | 🟡 MEDIUM |
| 6 | piano_pop_comping | 400 | 🟡 MEDIUM |
| 7 | strings_staccato | 368 | 🟡 MEDIUM |
| 8 | bass_walking | 352 | 🟡 MEDIUM |
| 9 | bass_slap | 200 | 🟢 LOW |
| 10 | piano_ballad | 200 | 🟢 LOW |

---

## 🏗️ Targets Strategy

### Real:Synthetic Ratios (楽器別調整)

**Guitar (70:30):**
- 理由: Strum/Arpeggio極端不足
- 合成優先度: strum > arpeggio > fingerpicking

**Bass (85:15):**
- 理由: 実データ高品質、Augmentation優先
- 合成優先度: walking > slap > humanization

**Strings (70:30):**
- 理由: Legato極端不足
- 合成優先度: **legato (757!) > spiccato > tremolo**

**Piano (90:10):**
- 理由: POP909高品質、最小限の合成
- 合成優先度: voicing variations > ballad

---

## 🔧 Quality Gates

### Synthetic Data Threshold (+5% stricter)

| 楽器 | Real閾値 | Synthetic閾値 | 合格率目標 |
|------|---------|--------------|-----------|
| Guitar | 40.0 | **45.0** | 75%+ |
| Bass | 40.0 | **45.0** | 80%+ |
| Strings | 45.0 | **50.0** | 75%+ |
| Piano | 45.0 | **50.0** | 95%+ |

### Data Source Priority

**Guitar:**
1. SLAKH real (70%)
2. Emotion synthetic (25%)
3. Suno improved (5%, max 100 files)

**Bass:**
1. SLAKH real (85%)
2. Augmentation (10% - timing jitter ±15ms)
3. Emotion synthetic (5%)

**Strings:**
1. SLAKH real (70%)
2. Emotion synthetic (25%)
3. Suno improved (5%, max 100 files)

**Piano:**
1. POP909 real (90%)
2. Emotion synthetic (10%)
3. Suno improved (0% - not used)

---

## 🚀 次フェーズ (実装予定)

### Phase 1: Manifest Runner実装

**scripts/run_manifest.py (ChatGPT framework):**
```python
# Manifest JSONL読み込み
# → Generator呼び出し (emotion_humanizer統合)
# → clean_midi適用
# → Shard pickle直書き (5,000/shard)
# → Resume対応 (既存shard検出)
```

**必要な統合:**
- `generator/{guitar,bass,strings,piano}_generator.py`
- `emotion_humanizer.py` (technique→params mapping)
- `scripts/cleaners/common.py` (LAMDA meta extraction)
- `scripts/shard_pickle_utils.py` (既存関数活用)

### Phase 2: 実行・検証

**小規模テスト (100 files):**
```bash
# Guitar strum 100件生成テスト
python scripts/run_manifest.py \
  --manifest manifests/test_guitar_strum_100.jsonl \
  --pickle-out data/shards/test \
  --shard-size 100 \
  --emit-midi-out audits/test
```

**品質検証:**
```bash
# Stage2で品質確認
python scripts/lamda_stage2_extractor.py \
  --metadata-index data/shards/test/INDEX.pkl \
  --input-dir audits/test \
  --config configs/lamda/guitar_stage2.yaml \
  --threshold 45.0
```

### Phase 3: 本番実行

**全manifest実行 (7,888 files):**
```bash
python scripts/run_manifest.py \
  --manifest manifests/manifest_20251017.jsonl \
  --pickle-out data/shards/hybrid \
  --shard-size 5000 \
  --seed 1234
```

**期待結果:**
- `data/shards/hybrid/guitar_shard_00000.pkl` ~ `00001.pkl`
- `data/shards/hybrid/strings_shard_00000.pkl`
- `data/shards/hybrid/bass_shard_00000.pkl`
- `data/shards/hybrid/piano_shard_00000.pkl`
- `data/shards/hybrid/INDEX.pkl`

### Phase 4: データ統合

**Real + Synthetic統合:**
```bash
python scripts/merge_real_synthetic.py \
  --real-shards output/{slakh,pop909}/shards \
  --synthetic-shards data/shards/hybrid \
  --output data/shards/integrated \
  --targets configs/targets_hybrid.yaml
```

**Stratified Split:**
```bash
python scripts/prepare_stratified_splits.py \
  --input data/shards/integrated \
  --output data/splits/hybrid \
  --strata technique,tempo,instrument \
  --ratio 80:10:10 \
  --synthetic-only-train on
```

---

## 📚 Integration with Existing Systems

### ChatGPT提案との整合性

| 項目 | ChatGPT提案 | 実装状況 |
|-----|-----------|---------|
| Manifest JSONL | ✅ 完全一致 | ✅ Done |
| Gap estimation | ✅ 完全一致 | ✅ Done |
| Shard pickle直書き | ✅ 提案通り | ⏳ 次フェーズ |
| Resume対応 | ✅ 提案通り | ⏳ 次フェーズ |
| Quality gate | ✅ +5%ルール | ✅ Done |

### 既存システムとの互換性

**✅ 互換:**
- `configs/lamda/*_stage2.yaml` (品質ゲート)
- `scripts/cleaners/common.py` (LAMDA meta)
- `output/*/shards/*.pkl` (既存shard構造)

**🔧 拡張必要:**
- `generator/*_generator.py` (technique params mapping)
- `emotion_humanizer.py` (manifest→emotion profile変換)

---

## 🎯 Success Metrics

### Phase 1 Success Criteria
- [ ] run_manifest.py実装完了
- [ ] 100ファイルテスト実行成功
- [ ] Stage2品質ゲート通過率 > 75%

### Phase 2 Success Criteria
- [ ] 7,888ファイル生成完了
- [ ] Shard pickle正常生成
- [ ] 品質ゲート通過 (各楽器)

### Phase 3 Success Criteria
- [ ] Real+Synthetic統合完了
- [ ] Stratified split生成
- [ ] Train/Val/Test正常分離

---

## 🔍 Risk Management

### Risk 1: Generator品質不足
- **対策**: 小規模テスト → 品質確認 → イテレーション
- **Fallback**: Augmentation比率増加 (Bassで実証済み)

### Risk 2: Manifest実行時間
- **見積**: 7,888 files × 1秒/file = ~2.2時間
- **対策**: 並列化 (--jobs 4 → ~30分)
- **Monitor**: Progress bar + ETA表示

### Risk 3: Shard pickle容量
- **見積**: 7,888 files × 50KB/file = ~394MB
- **対策**: Shard分割 (5,000/shard = 2 shards/楽器)
- **Monitor**: Disk usage tracking

---

## 📝 Documentation

**作成済み:**
- `docs/HYBRID_LEARNING_STRATEGY.md` (Phase 1-5 roadmap)
- `docs/HYBRID_STRATEGY_INTEGRATED.md` (統合戦略)
- `docs/STAGE2_FULL_PRODUCTION_REPORT.md` (実データ分析)
- `MULTI_DATASET_RUNNER_GUIDE.md` (共通ランナー)
- **このファイル** (Manifest実装レポート)

**次回作成:**
- `docs/MANIFEST_RUNNER_GUIDE.md` (run_manifest.py使用法)
- `docs/DATA_INTEGRATION_REPORT.md` (統合結果)

---

## 🏁 Conclusion

### 達成事項

1. ✅ **Technique Distribution Analysis完了**
   - 全4楽器の現状分布を定量化
   - Critical gaps特定 (guitar_strum 1,554, strings_legato 1,117)

2. ✅ **Manifest-Driven Architecture確立**
   - ChatGPT提案と完全統合
   - 7,888ファイルの合成計画策定
   - Real:Synthetic ratios最適化

3. ✅ **Quality Gates定義**
   - Synthetic +5%ルール適用
   - 楽器別閾値・合格率設定

### 次のマイルストーン

**Immediate (1 week):**
- run_manifest.py実装
- 100ファイルテスト実行

**Short-term (2 weeks):**
- 全manifest実行 (7,888 files)
- 品質検証

**Mid-term (1 month):**
- Real+Synthetic統合
- Training dataset準備

---

**Report Version**: 1.0  
**Generated**: 2025-10-17  
**Status**: ✅ Manifest Generation Complete, Ready for Phase 1 (Runner Implementation)
