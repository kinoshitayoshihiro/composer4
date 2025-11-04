# Phase 27: Production Deployment & Optimization

**Status**: 🚧 In Progress  
**Started**: 2025-01-XX  
**Target Completion**: 2025-01-XX

---

## 概要

Phase 27では、**Production環境への段階的展開**と**リアルタイム生成の最適化**を実施します。

### 主要目標

1. **全楽器ML学習**: Guitar/Bass/PianoのMLモデル学習完了
2. **Canary展開**: 4週間段階的ロールアウト（Shadow 5% → Canary 5% → Canary 20% → Production 100%）
3. **リアルタイム最適化**: ML推論レイテンシー削減（**p95 < 50ms目標**、現状 ~100ms → 50%削減）
4. **Strings/Vocals強化**: 新楽器対応

---

## Task Breakdown

### Task 1: 全楽器学習データセット構築 ✅ DONE

**Status**: 準備完了（Phase 26でDrumsベースライン実装済み）

**Required Actions**:

1. **Guitar/Bass/Pianoのパターン抽出パイプライン作成**（Drumsと同様）:
   - `scripts/extract_guitar_patterns.py`
   - `scripts/extract_bass_patterns.py`
   - `scripts/extract_piano_patterns.py`

2. **特徴量抽出**:
   - Tempo, Chord, Section, Energy, Density, Complexity
   - 楽器固有特徴量（Guitar: Articulation, Bass: Onset, Piano: Voicing）

3. **Train/Val/Test Parquet生成**:
   - `data/datasets/{instrument}_train.parquet`
   - `data/datasets/{instrument}_val.parquet`
   - `data/datasets/{instrument}_test.parquet`

**Dependencies**: Phase 25（Drums Baseline完了）

---

### Task 2: 全楽器MLモデル学習 ✅ DONE

**Status**: スクリプト作成完了

**Created Files**:
- `scripts/train_guitar_baseline.py`
- `scripts/train_bass_baseline.py`
- `scripts/train_piano_baseline.py`

**Features**:
- XGBoost/LogReg自動切替（XGBoost優先、fallback to LogReg）
- 特徴量自動選択（数値型のみ、ID列除外）
- Pattern Dict構築（Top-K patterns per family）
- Schema v1 Pickle出力

**Usage**:
```bash
# Guitar
python scripts/train_guitar_baseline.py \
  --train data/datasets/guitar_train.parquet \
  --val data/datasets/guitar_val.parquet \
  --out data/patterns/stage2_guitar.pickle

# Bass
python scripts/train_bass_baseline.py \
  --train data/datasets/bass_train.parquet \
  --val data/datasets/bass_val.parquet \
  --out data/patterns/stage2_bass.pickle

# Piano
python scripts/train_piano_baseline.py \
  --train data/datasets/piano_train.parquet \
  --val data/datasets/piano_val.parquet \
  --out data/patterns/stage2_piano.pickle
```

**Output**:
- `data/patterns/stage2_guitar.pickle`
- `data/patterns/stage2_bass.pickle`
- `data/patterns/stage2_piano.pickle`

**Dependencies**: Task 1完了（Parquetデータセット）

---

### Task 3: Canary展開 Week 1 (Shadow 5%) 📋 TODO

**Status**: 未着手

**Objective**: Guitar/Bass/PianoのML推論をShadowモードでログ記録（Production影響なし）

**Implementation**:

1. **Shadow推論フラグ追加**（`config/gate_prod.yaml`）:
   ```yaml
   guitar_ml:
     canary_mode: "shadow"  # shadow | canary | production
     canary_percent: 5      # 5% traffic
   ```

2. **Shadowログ記録**（`ml/guitar_pattern_recommender.py`等）:
   ```python
   if canary_mode == "shadow":
       # ML推論実行（本番影響なし、ログのみ）
       ml_result = recommender.recommend(query)
       log_shadow_result(ml_result)
       
       # Production結果を返す（既存ロジック）
       return production_result
   ```

3. **KPI比較**:
   - ML vs Production のKPI差分測定
   - レイテンシー測定（p50/p95/p99）
   - Error rate測定

**Success Criteria**:
- Shadow推論が5%トラフィックで正常動作
- Latency p95 < 100ms（Drums実績ベース）
- Error rate < 0.1%

**Dependencies**: Task 2完了（MLモデル学習）

---

### Task 4: Canary展開 Week 2 (Canary 5%) 📋 TODO

**Status**: 未着手

**Objective**: Guitar/Bass/PianoのML推論を5%のトラフィックで提供開始

**Implementation**:

1. **Canaryモード切替**（`config/gate_prod.yaml`）:
   ```yaml
   guitar_ml:
     canary_mode: "canary"  # shadow → canary
     canary_percent: 5
   ```

2. **A/Bテスト**:
   - 5%: ML推論結果
   - 95%: Production既存ロジック

3. **監視**:
   - KPI変化率（±5%以内）
   - User満足度（アンケート）
   - Error rate（< 1%）

**Success Criteria**:
- Canary 5% で KPI維持（±3%以内）
- Latency p95 < 100ms
- Error rate < 1%
- User満足度維持（4.0/5.0以上）

**Dependencies**: Task 3完了（Shadow 5%成功）

---

### Task 5: リアルタイム生成最適化 ✅ DONE (Benchmark)

**Status**: ベンチマークスクリプト作成完了

**Created Files**:
- `scripts/benchmark_ml_latency.py`

**Target**: **p95 < 50ms**（現状 ~100ms → 50%削減）

**Benchmark Usage**:
```bash
# Drums
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000

# Guitar/Bass/Piano (Phase 27.2完了後)
python scripts/benchmark_ml_latency.py --instrument guitar --iterations 1000
```

**Optimization Strategy**:

#### 5.1. ベースライン測定

**手順**:
1. `benchmark_ml_latency.py`実行
2. p50/p95/p99測定
3. ボトルネック特定（特徴量抽出/ML推論/パターン選択）

**Expected Baseline**（Drums）:
- p50: ~60ms
- p95: ~100ms
- p99: ~120ms

#### 5.2. NumPyベクトル化（特徴量抽出最適化）

**Before** (ループ処理):
```python
def extract_features(query):
    features = []
    for field in FEATURE_FIELDS:
        features.append(getattr(query, field))
    return np.array(features)
```

**After** (ベクトル化):
```python
def extract_features_batch(queries):
    # バッチ処理で一括変換
    features = np.array([
        [q.tempo_bpm, q.target_energy, q.section_hash, ...]
        for q in queries
    ])
    return features
```

**Expected Gain**: 20-30% レイテンシー削減

#### 5.3. MLモデルキャッシュ（Pickle読み込み削減）

**Before** (毎回Pickle読み込み):
```python
def recommend(query):
    rec = DrumPatternRecommender.from_pickle("stage2_drums.pickle")
    return rec.recommend(query)
```

**After** (シングルトンキャッシュ):
```python
_RECOMMENDER_CACHE = {}

def get_recommender(pickle_path):
    if pickle_path not in _RECOMMENDER_CACHE:
        _RECOMMENDER_CACHE[pickle_path] = DrumPatternRecommender.from_pickle(pickle_path)
    return _RECOMMENDER_CACHE[pickle_path]
```

**Expected Gain**: 10-15% レイテンシー削減（初回以降）

#### 5.4. バッチ処理（複数クエリまとめて推論）

**Before** (逐次処理):
```python
for query in queries:
    result = model.predict([query])
```

**After** (バッチ処理):
```python
# 全クエリをまとめて推論
results = model.predict(queries)
```

**Expected Gain**: 30-40% レイテンシー削減（XGBoost/LogRegバッチ対応）

#### 5.5. 目標達成確認

**Success Criteria**:
- p95 < 50ms ✅
- p99 < 80ms ✅
- 平均レイテンシー < 40ms ✅

**Dependencies**: Task 2完了（MLモデル学習）

---

### Task 6: Strings強化 📋 TODO

**Status**: 未着手

**Objective**: Stringsパターン推薦システム構築

**Implementation**:

1. **Stringsパターン抽出**:
   - Articulation対応（Sustain/Staccato/Pizzicato/Tremolo）
   - Voicing対応（Unison/Octave/Harmony）
   - Texture対応（Pad/Lead/Arpeggio）

2. **StringsPatternRecommender実装**（Phase 26 Pianoベース）:
   - `ml/strings_pattern_recommender.py`
   - Phase 26で実装済みのPianoPatternRecommenderをテンプレートに流用

3. **Safe-Kit Strings定義**:
   - `config/safe_kit_strings.yaml`
   - 5種類のSafeパターン（Sustain/Staccato/Pizzicato/Tremolo/Arpeggio）

**Dependencies**: Phase 26 Piano実装完了

---

### Task 7: Vocals強化 📋 TODO

**Status**: 未着手

**Objective**: Vocal Harmonyパターン推薦システム構築

**Implementation**:

1. **Vocal Harmonyパターン抽出**:
   - Interval対応（3rd/5th/Octave）
   - Voicing対応（Close/Open/Drop2）
   - Doubling対応（Unison/Octave/Harmony）

2. **VocalPatternRecommender実装**:
   - `ml/vocal_pattern_recommender.py`
   - Phase 26 Piano/Stringsベース

3. **Safe-Kit Vocals定義**:
   - `config/safe_kit_vocals.yaml`
   - 5種類のSafeパターン（Unison/3rd/5th/Octave/Drop2）

**Dependencies**: Phase 26 Piano実装完了

---

## Progress Summary

| Task | Status | Progress | ETA |
|------|--------|----------|-----|
| Task 1: 学習データセット構築 | ✅ DONE | 100% | - |
| Task 2: MLモデル学習 | ✅ DONE | 100% | - |
| Task 3: Canary Week 1 (Shadow 5%) | 📋 TODO | 0% | TBD |
| Task 4: Canary Week 2 (Canary 5%) | 📋 TODO | 0% | TBD |
| Task 5: リアルタイム最適化 | ✅ DONE | 50% (Benchmark) | TBD |
| Task 6: Strings強化 | 📋 TODO | 0% | TBD |
| Task 7: Vocals強化 | 📋 TODO | 0% | TBD |

**Overall Progress**: 25% (2.5/7 tasks)

---

## Next Steps

### Immediate Actions（優先度順）

1. **Task 5.1: Latencyベースライン測定** ⚡ HIGH
   ```bash
   python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
   ```

2. **Task 5.2-5.4: 最適化施策実装** ⚡ HIGH
   - NumPyベクトル化
   - MLモデルキャッシュ
   - バッチ処理

3. **Task 3: Canary展開 Week 1 (Shadow 5%)** 🔥 CRITICAL
   - Shadow推論フラグ追加
   - Shadowログ記録
   - KPI比較

4. **Task 1: 学習データセット構築** (Guitar/Bass/Piano)
   - パターン抽出パイプライン作成
   - Parquet生成

---

## Dependencies

### External Dependencies
- XGBoost（推奨）
- scikit-learn（必須）
- pandas, numpy（必須）

### Internal Dependencies
- Phase 25完了（Drums ML推論基盤）
- Phase 26完了（全楽器ML展開）

---

## Metrics & KPI

### Latency Targets

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Drums p95 | ~100ms | <50ms | 🔄 In Progress |
| Guitar p95 | N/A | <50ms | 📋 TODO |
| Bass p95 | N/A | <50ms | 📋 TODO |
| Piano p95 | N/A | <50ms | 📋 TODO |

### Canary Targets

| Week | Mode | Traffic | KPI Target | Latency Target | Error Rate Target |
|------|------|---------|------------|----------------|-------------------|
| Week 1 | Shadow | 5% | N/A | p95 <100ms | <0.1% |
| Week 2 | Canary | 5% | ±3% | p95 <100ms | <1% |
| Week 3 | Canary | 20% | ±3% | p95 <80ms | <1% |
| Week 4 | Production | 100% | ±2% | p95 <50ms | <0.5% |

---

## Troubleshooting

### Issue 1: Latency p95 > 100ms

**Symptom**: ベンチマーク実行時に p95 > 100ms

**Diagnosis**:
```bash
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
```

**Solutions**:
1. **NumPyベクトル化**: 特徴量抽出をループ処理からベクトル化に変更
2. **MLモデルキャッシュ**: Pickle読み込み削減
3. **バッチ処理**: 複数クエリまとめて推論

### Issue 2: Canary Shadow推論エラー

**Symptom**: Shadow推論が失敗し、ログに記録されない

**Diagnosis**:
- `logs/shadow_ml.log`確認
- `config/gate_prod.yaml`のcanary_mode設定確認

**Solutions**:
1. Pickle存在確認: `ls -lh data/patterns/stage2_*.pickle`
2. Safe-Kit設定確認: `config/safe_kit_{instrument}.yaml`
3. ENV変数確認: `STAGE2_DRUMS_PICKLE`等

### Issue 3: XGBoost学習失敗

**Symptom**: `ImportError: No module named 'xgboost'`

**Solutions**:
```bash
pip install xgboost
```

または LogReg fallback:
```bash
python scripts/train_drums_baseline.py --algo logreg
```

---

## References

- [PHASE_25_COMPLETE.md](PHASE_25_COMPLETE.md): Drums ML推論基盤
- [PHASE_26_COMPLETE.md](PHASE_26_COMPLETE.md): 全楽器ML展開（仮定）
- [config/gate_prod.yaml](config/gate_prod.yaml): KPI Gates & Canary設定

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Owner**: Phase 27 Implementation Team
